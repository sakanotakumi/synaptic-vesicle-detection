"""
Peak detection and Non-Maximum Suppression (NMS) functions.
"""

from typing import Tuple, List
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_maxima


def detect_peaks_3d(
    response: np.ndarray,
    threshold_percentile: float = 99.5,
    min_distance: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima in 3D response volume.
    
    Args:
        response: 3D response volume
        threshold_percentile: Percentile threshold for peak detection
        min_distance: Minimum distance between peaks
        
    Returns:
        peak_coords: Coordinates of detected peaks (N, 3)
        peak_values: Response values at peak locations
    """
    # Calculate threshold
    threshold = np.percentile(response, threshold_percentile)
    
    # Find local maxima
    peak_coords = peak_local_maxima(
        response,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=True
    )
    
    if len(peak_coords) == 0:
        return np.empty((0, 3)), np.empty(0)
    
    # Convert to array format
    peak_coords = np.array(peak_coords)
    
    # Get response values at peak locations
    peak_values = response[tuple(peak_coords.T)]
    
    return peak_coords, peak_values


def estimate_scale_at_peaks(
    peak_coords: np.ndarray,
    best_scale_idx: np.ndarray,
    sigma_list: np.ndarray
) -> np.ndarray:
    """
    Estimate optimal scale (radius) at each peak location.
    
    Args:
        peak_coords: Peak coordinates (N, 3)
        best_scale_idx: Best scale index volume
        sigma_list: Array of sigma values used
        
    Returns:
        estimated_radii: Estimated radii in voxels for each peak
    """
    if len(peak_coords) == 0:
        return np.empty(0)
    
    # Get scale indices at peak locations
    scale_indices = best_scale_idx[tuple(peak_coords.T)]
    
    # Convert sigma to radius: r ≈ √3 * σ
    estimated_sigma = sigma_list[scale_indices]
    estimated_radii = np.sqrt(3) * estimated_sigma
    
    return estimated_radii


def non_maximum_suppression(
    peak_coords: np.ndarray,
    peak_values: np.ndarray,
    estimated_radii: np.ndarray,
    nms_factor: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        peak_coords: Peak coordinates (N, 3)
        peak_values: Response values at peaks
        estimated_radii: Estimated radii for each peak
        nms_factor: Factor for NMS radius calculation
        
    Returns:
        filtered_coords: Coordinates after NMS
        filtered_values: Values after NMS
        filtered_radii: Radii after NMS
    """
    if len(peak_coords) == 0:
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    
    # Sort by response value (descending)
    sort_idx = np.argsort(peak_values)[::-1]
    sorted_coords = peak_coords[sort_idx]
    sorted_values = peak_values[sort_idx]
    sorted_radii = estimated_radii[sort_idx]
    
    # Keep track of which peaks to keep
    keep = np.ones(len(sorted_coords), dtype=bool)
    
    for i in range(len(sorted_coords)):
        if not keep[i]:
            continue
            
        # Calculate NMS radius for current peak
        nms_radius = nms_factor * sorted_radii[i]
        
        # Find other peaks within NMS radius
        distances = np.linalg.norm(
            sorted_coords[i+1:] - sorted_coords[i], axis=1
        )
        
        # Suppress nearby peaks with lower scores
        suppress_mask = distances < nms_radius
        keep[i+1:][suppress_mask] = False
    
    return sorted_coords[keep], sorted_values[keep], sorted_radii[keep]


def tile_boundary_nms(
    detections_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    tile_coords_list: List[Tuple[slice, slice, slice]],
    halo_size: int,
    nms_factor: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply NMS across tile boundaries.
    
    Args:
        detections_list: List of (coords, values, radii) for each tile
        tile_coords_list: List of tile slice coordinates
        halo_size: Halo size used in tiling
        nms_factor: Factor for NMS radius calculation
        
    Returns:
        global_coords: Global coordinates after boundary NMS
        global_values: Values after boundary NMS
        global_radii: Radii after boundary NMS
    """
    # Convert tile-local coordinates to global coordinates
    all_coords = []
    all_values = []
    all_radii = []
    all_tile_ids = []
    
    for tile_id, ((coords, values, radii), tile_slice) in enumerate(
        zip(detections_list, tile_coords_list)
    ):
        if len(coords) == 0:
            continue
            
        # Convert to global coordinates
        z_offset = tile_slice[0].start
        y_offset = tile_slice[1].start  
        x_offset = tile_slice[2].start
        
        global_coords_tile = coords + np.array([z_offset, y_offset, x_offset])
        
        # Only keep detections not in halo region (to avoid duplicates)
        z_min, z_max = halo_size, coords.shape[0] + tile_slice[0].start - tile_slice[0].stop + halo_size
        y_min, y_max = halo_size, coords.shape[0] + tile_slice[1].start - tile_slice[1].stop + halo_size  
        x_min, x_max = halo_size, coords.shape[0] + tile_slice[2].start - tile_slice[2].stop + halo_size
        
        # Filter out halo detections for interior tiles
        valid_mask = (
            (coords[:, 0] >= z_min) & (coords[:, 0] < z_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] < y_max) &
            (coords[:, 2] >= x_min) & (coords[:, 2] < x_max)
        )
        
        all_coords.append(global_coords_tile[valid_mask])
        all_values.append(values[valid_mask])
        all_radii.append(radii[valid_mask])
        all_tile_ids.extend([tile_id] * np.sum(valid_mask))
    
    if not all_coords:
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    
    # Concatenate all detections
    global_coords = np.vstack(all_coords)
    global_values = np.concatenate(all_values)
    global_radii = np.concatenate(all_radii)
    
    # Apply global NMS
    final_coords, final_values, final_radii = non_maximum_suppression(
        global_coords, global_values, global_radii, nms_factor
    )
    
    return final_coords, final_values, final_radii


def filter_by_top_k_per_volume(
    peak_coords: np.ndarray,
    peak_values: np.ndarray,
    estimated_radii: np.ndarray,
    volume_shape: Tuple[int, int, int],
    max_per_million_voxels: float = 100.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter detections to keep only top-K per volume.
    
    Args:
        peak_coords: Peak coordinates
        peak_values: Peak values
        estimated_radii: Estimated radii
        volume_shape: Shape of the volume
        max_per_million_voxels: Maximum detections per million voxels
        
    Returns:
        Filtered coordinates, values, and radii
    """
    if len(peak_coords) == 0:
        return peak_coords, peak_values, estimated_radii
    
    # Calculate maximum number of detections
    total_voxels = np.prod(volume_shape)
    max_detections = int(max_per_million_voxels * total_voxels / 1e6)
    
    if len(peak_coords) <= max_detections:
        return peak_coords, peak_values, estimated_radii
    
    # Sort by value and keep top-K
    sort_idx = np.argsort(peak_values)[::-1][:max_detections]
    
    return peak_coords[sort_idx], peak_values[sort_idx], estimated_radii[sort_idx]
