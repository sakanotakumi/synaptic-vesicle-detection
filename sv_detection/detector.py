"""
Main SV detector class that orchestrates the detection pipeline.
"""

import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .config import SVDetectionConfig
from .utils import (
    load_volume, load_mask, save_results, create_tile_iterator, 
    setup_output_directory, check_gpu_availability
)
from .preprocessing import (
    normalize_intensity_tile, apply_mask_to_volume, multiscale_response
)
from .peak_detection import (
    detect_peaks_3d, estimate_scale_at_peaks, non_maximum_suppression,
    tile_boundary_nms, filter_by_top_k_per_volume
)
from .features import compute_ringness, compute_hessian_eigenvalues, compute_integrated_score
from .qc import create_detection_summary, save_detection_cutouts, create_qc_plots


class SVDetector:
    """
    Main class for automatic synaptic vesicle detection.
    """
    
    def __init__(self, config: SVDetectionConfig):
        """
        Initialize the SV detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        config.validate()
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Check GPU availability
        self.use_gpu = config.use_gpu and check_gpu_availability()
        if config.use_gpu and not self.use_gpu:
            print("Warning: GPU requested but not available, falling back to CPU")
    
    def detect(
        self,
        volume_path: str,
        mask_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run SV detection on a volume.
        
        Args:
            volume_path: Path to input volume
            mask_path: Optional path to binary mask
            output_dir: Optional output directory for results
            
        Returns:
            DataFrame with detection results
        """
        start_time = time.time()
        
        print(f"Loading volume from {volume_path}")
        volume, volume_metadata = load_volume(volume_path)
        
        # Load mask if provided
        mask = None
        if mask_path:
            print(f"Loading mask from {mask_path}")
            mask = load_mask(mask_path, volume.shape)
        
        print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
        print(f"Using GPU: {self.use_gpu}")
        
        # Create output directory if specified
        if output_dir:
            output_path = setup_output_directory(output_dir)
        else:
            output_path = None
        
        # Process volume
        if volume.nbytes > self.config.chunk_size_mb * 1024**2:
            print("Large volume detected, using tiled processing")
            detections = self._detect_tiled(volume, mask)
        else:
            print("Processing volume in single chunk")
            detections = self._detect_single_tile(volume, mask)
        
        # Create results DataFrame
        results_df = self._create_results_dataframe(detections, volume_metadata)
        
        processing_time = time.time() - start_time
        print(f"Detection completed in {processing_time:.2f} seconds")
        print(f"Found {len(results_df)} SVs")
        
        # Save results if output directory specified
        if output_path:
            self._save_all_outputs(results_df, volume, output_path, processing_time)
        
        return results_df
    
    def _detect_single_tile(
        self, 
        volume: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect SVs in a single volume tile.
        
        Args:
            volume: Input volume
            mask: Optional binary mask
            
        Returns:
            Dictionary with detection results
        """
        # Preprocessing
        volume_norm = normalize_intensity_tile(
            volume, 
            self.config.intensity_percentiles,
            self.config.normalization_method
        )
        
        if mask is not None:
            volume_norm = apply_mask_to_volume(volume_norm, mask)
        
        # Multiscale response computation
        sigma_list = self.config.get_sigma_list()
        print(f"Computing multiscale response with sigmas: {sigma_list}")
        
        max_response, best_scale_idx = multiscale_response(
            volume_norm,
            sigma_list,
            use_dog=self.config.use_dog,
            use_gpu=self.use_gpu
        )
        
        # Peak detection
        peak_coords, peak_values = detect_peaks_3d(
            max_response,
            self.config.peak_threshold_percentile,
            self.config.min_distance_voxel
        )
        
        if len(peak_coords) == 0:
            return {
                "coords": np.empty((0, 3)),
                "log_responses": np.empty(0),
                "radii": np.empty(0),
                "ringness": np.empty(0),
                "isotropy": np.empty(0),
                "scores": np.empty(0)
            }
        
        # Estimate scales
        estimated_radii = estimate_scale_at_peaks(peak_coords, best_scale_idx, sigma_list)
        
        # NMS
        nms_coords, nms_values, nms_radii = non_maximum_suppression(
            peak_coords, peak_values, estimated_radii, self.config.nms_radius_factor
        )
        
        if len(nms_coords) == 0:
            return {
                "coords": np.empty((0, 3)),
                "log_responses": np.empty(0),
                "radii": np.empty(0),
                "ringness": np.empty(0),
                "isotropy": np.empty(0),
                "scores": np.empty(0)
            }
        
        # Feature computation
        print(f"Computing features for {len(nms_coords)} candidates")
        
        ringness_scores = compute_ringness(
            volume_norm,
            nms_coords,
            nms_radii,
            self.config.ringness_inner_offset,
            self.config.ringness_outer_thickness,
            self.config.ringness_robustness_steps
        )
        
        isotropy_scores, _ = compute_hessian_eigenvalues(
            volume_norm,
            nms_coords,
            nms_radii,
            self.config.hessian_sigma_factor
        )
        
        # Integrated scoring
        integrated_scores = compute_integrated_score(
            nms_values,
            ringness_scores,
            isotropy_scores,
            self.config.score_weights
        )
        
        # Final filtering
        if self.config.score_threshold is not None:
            score_mask = integrated_scores >= self.config.score_threshold
        else:
            # Auto-determine threshold or use top-K
            final_coords, final_scores, final_radii = filter_by_top_k_per_volume(
                nms_coords, integrated_scores, nms_radii,
                volume.shape, self.config.max_detections_per_volume_voxel
            )
            
            # Apply the filtering to other arrays
            score_mask = np.isin(integrated_scores, final_scores)
        
        return {
            "coords": nms_coords[score_mask],
            "log_responses": nms_values[score_mask],
            "radii": nms_radii[score_mask],
            "ringness": ringness_scores[score_mask],
            "isotropy": isotropy_scores[score_mask],
            "scores": integrated_scores[score_mask]
        }
    
    def _detect_tiled(
        self, 
        volume: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect SVs using tiled processing for large volumes.
        
        Args:
            volume: Input volume
            mask: Optional binary mask
            
        Returns:
            Dictionary with detection results
        """
        # Create tiles
        tiles = create_tile_iterator(
            volume.shape, 
            self.config.tile_size, 
            self.config.halo_size
        )
        
        print(f"Processing {len(tiles)} tiles of size {self.config.tile_size}^3")
        
        # Process tiles
        if self.config.n_workers == 1:
            # Sequential processing
            tile_results = []
            for i, tile_slice in enumerate(tiles):
                print(f"Processing tile {i+1}/{len(tiles)}")
                result = self._process_tile(volume, mask, tile_slice)
                tile_results.append(result)
        else:
            # Parallel processing
            n_workers = self.config.n_workers
            if n_workers == -1:
                n_workers = mp.cpu_count()
            
            print(f"Using {n_workers} workers for parallel processing")
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                tile_results = list(executor.map(
                    self._process_tile_wrapper,
                    [(volume, mask, tile_slice, self.config) for tile_slice in tiles]
                ))
        
        # Combine results from all tiles
        all_detections = []
        for tile_result in tile_results:
            if len(tile_result["coords"]) > 0:
                all_detections.append((
                    tile_result["coords"],
                    tile_result["scores"], 
                    tile_result["radii"]
                ))
        
        if not all_detections:
            return {
                "coords": np.empty((0, 3)),
                "log_responses": np.empty(0),
                "radii": np.empty(0),
                "ringness": np.empty(0),
                "isotropy": np.empty(0),
                "scores": np.empty(0)
            }
        
        # Apply boundary NMS
        print("Applying boundary NMS")
        final_coords, final_scores, final_radii = tile_boundary_nms(
            all_detections, tiles, self.config.halo_size, self.config.nms_radius_factor
        )
        
        # Re-compute features for final detections (simplified for now)
        # In a full implementation, you might want to store and combine features from tiles
        n_final = len(final_coords)
        return {
            "coords": final_coords,
            "log_responses": final_scores,  # Using scores as proxy
            "radii": final_radii,
            "ringness": np.zeros(n_final),  # Would need to recompute
            "isotropy": np.zeros(n_final),  # Would need to recompute  
            "scores": final_scores
        }
    
    def _process_tile(
        self, 
        volume: np.ndarray, 
        mask: Optional[np.ndarray], 
        tile_slice: Tuple[slice, slice, slice]
    ) -> Dict[str, np.ndarray]:
        """
        Process a single tile.
        
        Args:
            volume: Full volume
            mask: Full mask (optional)
            tile_slice: Slice for this tile
            
        Returns:
            Dictionary with tile detection results
        """
        # Extract tile
        tile_volume = volume[tile_slice]
        tile_mask = mask[tile_slice] if mask is not None else None
        
        # Detect in tile
        return self._detect_single_tile(tile_volume, tile_mask)
    
    @staticmethod
    def _process_tile_wrapper(args):
        """Wrapper for multiprocessing."""
        volume, mask, tile_slice, config = args
        detector = SVDetector(config)
        return detector._process_tile(volume, mask, tile_slice)
    
    def _create_results_dataframe(
        self, 
        detections: Dict[str, np.ndarray],
        volume_metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create results DataFrame from detection dictionary.
        
        Args:
            detections: Detection results dictionary
            volume_metadata: Volume metadata
            
        Returns:
            DataFrame with standardized columns
        """
        if len(detections["coords"]) == 0:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'id', 'z_vox', 'y_vox', 'x_vox', 'z_um', 'y_um', 'x_um',
                'r_vox', 'r_nm', 'score', 'tile_id'
            ])
        
        coords = detections["coords"]
        radii = detections["radii"]
        scores = detections["scores"]
        
        # Create DataFrame
        df = pd.DataFrame({
            'id': range(len(coords)),
            'z_vox': coords[:, 0],
            'y_vox': coords[:, 1], 
            'x_vox': coords[:, 2],
            'z_um': coords[:, 0] * self.config.voxel_size_nm[0] / 1000,
            'y_um': coords[:, 1] * self.config.voxel_size_nm[1] / 1000,
            'x_um': coords[:, 2] * self.config.voxel_size_nm[2] / 1000,
            'r_vox': radii,
            'r_nm': radii * np.mean(self.config.voxel_size_nm),  # Average voxel size
            'score': scores,
            'tile_id': 0  # Would be properly set in tiled processing
        })
        
        return df
    
    def _save_all_outputs(
        self,
        results_df: pd.DataFrame,
        volume: np.ndarray,
        output_path: Path,
        processing_time: float
    ) -> None:
        """
        Save all outputs including results, QC, and metadata.
        
        Args:
            results_df: Detection results
            volume: Original volume
            output_path: Output directory
            processing_time: Processing time in seconds
        """
        print(f"Saving results to {output_path}")
        
        # Save main results
        config_dict = self.config.__dict__.copy()
        saved_files = save_results(
            results_df, output_path, config_dict, self.config.output_formats
        )
        
        # Create summary
        summary = create_detection_summary(results_df, processing_time, config_dict)
        
        # Save metrics
        from .utils import save_metrics
        metrics_path = output_path / "metrics.jsonl"
        save_metrics([summary], metrics_path)
        
        # Generate QC outputs if requested
        if self.config.save_qc_images and len(results_df) > 0:
            print("Generating QC plots and cutouts")
            
            # QC plots
            qc_plots = create_qc_plots(results_df, output_path, config_dict)
            
            # Sample cutouts
            cutouts = save_detection_cutouts(
                volume, results_df, output_path, 
                voxel_size_nm=self.config.voxel_size_nm
            )
            
            print(f"Saved {len(qc_plots)} QC plots and {len(cutouts)} cutouts")
        
        print("All outputs saved successfully")
