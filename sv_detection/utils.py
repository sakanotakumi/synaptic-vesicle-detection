"""
Utility functions for SV detection.
"""

import os
import json
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import zarr
from skimage import io
import tifffile


def load_volume(
    path: Union[str, Path], 
    as_zarr: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a 3D volume from various formats.
    
    Args:
        path: Path to volume file (TIFF stack, Zarr, etc.)
        as_zarr: Whether to load as zarr array for memory efficiency
        
    Returns:
        volume: 3D numpy array or zarr array
        metadata: Dictionary with volume metadata
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {path}")
    
    metadata = {"path": str(path), "format": None}
    
    if path.suffix.lower() in ['.tif', '.tiff']:
        # Load TIFF stack
        if as_zarr:
            volume = zarr.open(tifffile.imread, path, mode='r')
        else:
            volume = tifffile.imread(path)
        metadata["format"] = "tiff"
        
    elif path.suffix.lower() == '.zarr' or path.is_dir():
        # Load Zarr array
        volume = zarr.open(path, mode='r')
        if not as_zarr:
            volume = np.array(volume)
        metadata["format"] = "zarr"
        
    else:
        # Try generic image loading
        volume = io.imread(path)
        metadata["format"] = "generic"
    
    # Ensure 3D
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    elif volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D")
    
    metadata.update({
        "shape": volume.shape,
        "dtype": str(volume.dtype),
        "size_mb": np.prod(volume.shape) * np.dtype(volume.dtype).itemsize / (1024**2)
    })
    
    return volume, metadata


def load_mask(
    path: Union[str, Path], 
    volume_shape: Tuple[int, int, int]
) -> Optional[np.ndarray]:
    """
    Load a binary mask for SV detection.
    
    Args:
        path: Path to mask file
        volume_shape: Expected shape of the mask
        
    Returns:
        mask: Binary 3D array or None if path is None
    """
    if path is None:
        return None
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    
    if path.suffix.lower() in ['.tif', '.tiff']:
        mask = tifffile.imread(path)
    elif path.suffix.lower() == '.zarr' or path.is_dir():
        mask = np.array(zarr.open(path, mode='r'))
    else:
        mask = io.imread(path)
    
    # Ensure 3D
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    elif mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")
    
    # Check shape
    if mask.shape != volume_shape:
        raise ValueError(f"Mask shape {mask.shape} doesn't match volume shape {volume_shape}")
    
    # Convert to boolean
    return mask.astype(bool)


def save_results(
    detections: pd.DataFrame,
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    formats: List[str] = ["parquet", "napari_csv"]
) -> Dict[str, str]:
    """
    Save detection results in various formats.
    
    Args:
        detections: DataFrame with detection results
        output_dir: Output directory
        config: Configuration dictionary
        formats: List of output formats
        
    Returns:
        Dictionary mapping format names to output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save as Parquet
    if "parquet" in formats:
        parquet_path = output_dir / "sv_points.parquet"
        detections.to_parquet(parquet_path, index=False)
        saved_files["parquet"] = str(parquet_path)
    
    # Save as CSV
    if "csv" in formats:
        csv_path = output_dir / "sv_points.csv"
        detections.to_csv(csv_path, index=False)
        saved_files["csv"] = str(csv_path)
    
    # Save Napari-compatible CSV
    if "napari_csv" in formats:
        napari_df = create_napari_points_csv(detections)
        napari_path = output_dir / "sv_points_napari.csv"
        napari_df.to_csv(napari_path, index=False)
        saved_files["napari_csv"] = str(napari_path)
    
    # Save configuration
    config_path = output_dir / "run_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    saved_files["config"] = str(config_path)
    
    return saved_files


def create_napari_points_csv(detections: pd.DataFrame) -> pd.DataFrame:
    """
    Convert detection results to Napari Points format.
    
    Args:
        detections: DataFrame with detection results
        
    Returns:
        DataFrame in Napari Points format
    """
    napari_df = pd.DataFrame()
    
    # Napari expects z, y, x coordinates
    napari_df['z'] = detections['z_vox']
    napari_df['y'] = detections['y_vox'] 
    napari_df['x'] = detections['x_vox']
    
    # Add properties
    for col in ['r_vox', 'r_nm', 'score', 'tile_id']:
        if col in detections.columns:
            napari_df[col] = detections[col]
    
    return napari_df


def save_metrics(
    metrics: List[Dict[str, Any]], 
    output_path: Union[str, Path]
) -> None:
    """
    Save processing metrics to JSONL file.
    
    Args:
        metrics: List of metric dictionaries
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')


def create_tile_iterator(
    volume_shape: Tuple[int, int, int],
    tile_size: int,
    halo_size: int = 0
) -> List[Tuple[slice, ...]]:
    """
    Create tile slices for processing large volumes.
    
    Args:
        volume_shape: Shape of the full volume (z, y, x)
        tile_size: Size of each tile
        halo_size: Overlap between tiles
        
    Returns:
        List of slice tuples for each tile
    """
    tiles = []
    z_max, y_max, x_max = volume_shape
    
    for z_start in range(0, z_max, tile_size - 2*halo_size):
        for y_start in range(0, y_max, tile_size - 2*halo_size):
            for x_start in range(0, x_max, tile_size - 2*halo_size):
                
                z_end = min(z_start + tile_size, z_max)
                y_end = min(y_start + tile_size, y_max)
                x_end = min(x_start + tile_size, x_max)
                
                # Adjust start positions for halo
                z_start_halo = max(0, z_start - halo_size)
                y_start_halo = max(0, y_start - halo_size)
                x_start_halo = max(0, x_start - halo_size)
                
                # Adjust end positions for halo
                z_end_halo = min(z_max, z_end + halo_size)
                y_end_halo = min(y_max, y_end + halo_size)
                x_end_halo = min(x_max, x_end + halo_size)
                
                tile_slice = (
                    slice(z_start_halo, z_end_halo),
                    slice(y_start_halo, y_end_halo),
                    slice(x_start_halo, x_end_halo)
                )
                
                tiles.append(tile_slice)
    
    return tiles


def normalize_intensity(
    volume: np.ndarray,
    percentiles: Tuple[float, float] = (2.0, 98.0),
    method: str = "z_score"
) -> np.ndarray:
    """
    Normalize volume intensity.
    
    Args:
        volume: Input volume
        percentiles: Percentile range for clipping
        method: Normalization method ("z_score" or "min_max")
        
    Returns:
        Normalized volume
    """
    # Clip outliers
    p_low, p_high = np.percentile(volume, percentiles)
    volume_clipped = np.clip(volume, p_low, p_high)
    
    if method == "z_score":
        mean_val = np.mean(volume_clipped)
        std_val = np.std(volume_clipped)
        if std_val > 0:
            return (volume_clipped - mean_val) / std_val
        else:
            return volume_clipped - mean_val
            
    elif method == "min_max":
        min_val = np.min(volume_clipped)
        max_val = np.max(volume_clipped)
        if max_val > min_val:
            return (volume_clipped - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(volume_clipped)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_gpu_availability() -> bool:
    """Check if GPU acceleration is available."""
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
        return True
    except (ImportError, Exception):
        return False


def setup_output_directory(output_dir: Union[str, Path]) -> Path:
    """
    Setup output directory structure.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to created output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "qc_images").mkdir(exist_ok=True)
    (output_dir / "intermediate").mkdir(exist_ok=True)
    
    return output_dir
