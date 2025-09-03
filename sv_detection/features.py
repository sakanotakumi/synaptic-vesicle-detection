"""
Feature extraction functions for SV characterization.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage
from numba import jit
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@jit(nopython=True)
def compute_sphere_mask_indices(radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D indices for a sphere mask using Numba for speed.
    
    Args:
        radius: Sphere radius in voxels
        
    Returns:
        z, y, x indices for sphere voxels
    """
    r_int = int(np.ceil(radius))
    indices_z = []
    indices_y = []
    indices_x = []
    
    for z in range(-r_int, r_int + 1):
        for y in range(-r_int, r_int + 1):
            for x in range(-r_int, r_int + 1):
                if z*z + y*y + x*x <= radius*radius:
                    indices_z.append(z)
                    indices_y.append(y)
                    indices_x.append(x)
    
    return np.array(indices_z), np.array(indices_y), np.array(indices_x)


@jit(nopython=True)
def compute_annulus_mask_indices(
    inner_radius: float, 
    outer_radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D indices for an annulus (shell) mask.
    
    Args:
        inner_radius: Inner radius of annulus
        outer_radius: Outer radius of annulus
        
    Returns:
        z, y, x indices for annulus voxels
    """
    r_int = int(np.ceil(outer_radius))
    indices_z = []
    indices_y = []
    indices_x = []
    
    for z in range(-r_int, r_int + 1):
        for y in range(-r_int, r_int + 1):
            for x in range(-r_int, r_int + 1):
                dist_sq = z*z + y*y + x*x
                if inner_radius*inner_radius <= dist_sq <= outer_radius*outer_radius:
                    indices_z.append(z)
                    indices_y.append(y)
                    indices_x.append(x)
    
    return np.array(indices_z), np.array(indices_y), np.array(indices_x)


def compute_ringness(
    volume: np.ndarray,
    peak_coords: np.ndarray,
    estimated_radii: np.ndarray,
    inner_offset: int = 2,
    outer_thickness: int = 2,
    robustness_steps: int = 3
) -> np.ndarray:
    """
    Compute ringness feature for detected peaks.
    Ringness = mean(annulus) - mean(inner_sphere)
    Negative values indicate dark shell with bright center (SV-like).
    
    Args:
        volume: Input volume
        peak_coords: Peak coordinates (N, 3)
        estimated_radii: Estimated radii for each peak
        inner_offset: Offset to reduce inner sphere radius
        outer_thickness: Thickness of outer annulus
        robustness_steps: Number of radius variations for robustness
        
    Returns:
        ringness_scores: Ringness values for each peak
    """
    if len(peak_coords) == 0:
        return np.empty(0)
    
    ringness_scores = np.zeros(len(peak_coords))
    
    for i, (coord, radius) in enumerate(zip(peak_coords, estimated_radii)):
        z, y, x = coord
        
        # Try multiple radius variations for robustness
        ringness_values = []
        
        for offset in range(-robustness_steps//2 + 1, robustness_steps//2 + 1):
            adj_radius = max(1.0, radius + offset)
            
            # Define inner sphere and outer annulus
            inner_radius = max(0.5, adj_radius - inner_offset)
            outer_radius_inner = adj_radius
            outer_radius_outer = adj_radius + outer_thickness
            
            # Get sphere indices
            inner_z, inner_y, inner_x = compute_sphere_mask_indices(inner_radius)
            annulus_z, annulus_y, annulus_x = compute_annulus_mask_indices(
                outer_radius_inner, outer_radius_outer
            )
            
            # Convert to global coordinates
            inner_coords_z = inner_z + z
            inner_coords_y = inner_y + y
            inner_coords_x = inner_x + x
            
            annulus_coords_z = annulus_z + z
            annulus_coords_y = annulus_y + y
            annulus_coords_x = annulus_x + x
            
            # Check bounds
            valid_inner = (
                (inner_coords_z >= 0) & (inner_coords_z < volume.shape[0]) &
                (inner_coords_y >= 0) & (inner_coords_y < volume.shape[1]) &
                (inner_coords_x >= 0) & (inner_coords_x < volume.shape[2])
            )
            
            valid_annulus = (
                (annulus_coords_z >= 0) & (annulus_coords_z < volume.shape[0]) &
                (annulus_coords_y >= 0) & (annulus_coords_y < volume.shape[1]) &
                (annulus_coords_x >= 0) & (annulus_coords_x < volume.shape[2])
            )
            
            if np.sum(valid_inner) == 0 or np.sum(valid_annulus) == 0:
                continue
            
            # Extract values
            inner_values = volume[
                inner_coords_z[valid_inner],
                inner_coords_y[valid_inner], 
                inner_coords_x[valid_inner]
            ]
            
            annulus_values = volume[
                annulus_coords_z[valid_annulus],
                annulus_coords_y[valid_annulus],
                annulus_coords_x[valid_annulus]
            ]
            
            # Compute ringness
            mean_inner = np.mean(inner_values)
            mean_annulus = np.mean(annulus_values)
            ringness = mean_annulus - mean_inner
            
            ringness_values.append(ringness)
        
        # Take minimum (most negative) ringness for robustness
        if ringness_values:
            ringness_scores[i] = np.min(ringness_values)
        else:
            ringness_scores[i] = 0.0
    
    return ringness_scores


def compute_hessian_eigenvalues(
    volume: np.ndarray,
    peak_coords: np.ndarray,
    estimated_radii: np.ndarray,
    sigma_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hessian-based sphericity features.
    
    Args:
        volume: Input volume
        peak_coords: Peak coordinates (N, 3)
        estimated_radii: Estimated radii for each peak
        sigma_factor: Factor to multiply estimated radius for Gaussian smoothing
        
    Returns:
        isotropy_scores: Isotropy scores (|λ3|/|λ1|)
        eigenvalue_sums: Sum of eigenvalues (trace of Hessian)
    """
    if len(peak_coords) == 0:
        return np.empty(0), np.empty(0)
    
    isotropy_scores = np.zeros(len(peak_coords))
    eigenvalue_sums = np.zeros(len(peak_coords))
    
    # Pre-compute smoothed volume
    # Use average radius for global smoothing
    avg_radius = np.mean(estimated_radii)
    sigma = avg_radius * sigma_factor / np.sqrt(3)  # Convert radius to sigma
    smoothed_volume = ndimage.gaussian_filter(volume, sigma)
    
    # Compute Hessian components
    hxx = ndimage.sobel(ndimage.sobel(smoothed_volume, axis=2), axis=2)
    hyy = ndimage.sobel(ndimage.sobel(smoothed_volume, axis=1), axis=1)
    hzz = ndimage.sobel(ndimage.sobel(smoothed_volume, axis=0), axis=0)
    hxy = ndimage.sobel(ndimage.sobel(smoothed_volume, axis=2), axis=1)
    hxz = ndimage.sobel(ndimage.sobel(smoothed_volume, axis=2), axis=0)
    hyz = ndimage.sobel(ndimage.sobel(smoothed_volume, axis=1), axis=0)
    
    for i, coord in enumerate(peak_coords):
        z, y, x = coord.astype(int)
        
        # Check bounds
        if (z < 0 or z >= volume.shape[0] or
            y < 0 or y >= volume.shape[1] or  
            x < 0 or x >= volume.shape[2]):
            continue
        
        # Build Hessian matrix at this point
        hessian = np.array([
            [hzz[z, y, x], hyz[z, y, x], hxz[z, y, x]],
            [hyz[z, y, x], hyy[z, y, x], hxy[z, y, x]],
            [hxz[z, y, x], hxy[z, y, x], hxx[z, y, x]]
        ])
        
        try:
            # Compute eigenvalues
            eigenvals = np.linalg.eigvals(hessian)
            eigenvals_abs = np.abs(eigenvals)
            eigenvals_abs.sort()  # Sort in ascending order
            
            # Isotropy: ratio of smallest to largest eigenvalue
            if eigenvals_abs[2] > 1e-10:
                isotropy = eigenvals_abs[0] / eigenvals_abs[2]
            else:
                isotropy = 0.0
            
            isotropy_scores[i] = isotropy
            eigenvalue_sums[i] = np.sum(eigenvals)  # Trace (with sign)
            
        except np.linalg.LinAlgError:
            isotropy_scores[i] = 0.0
            eigenvalue_sums[i] = 0.0
    
    return isotropy_scores, eigenvalue_sums


def compute_integrated_score(
    log_responses: np.ndarray,
    ringness_scores: np.ndarray,
    isotropy_scores: np.ndarray,
    weights: dict = None
) -> np.ndarray:
    """
    Compute integrated detection score from multiple features.
    
    Args:
        log_responses: LoG response values
        ringness_scores: Ringness feature values
        isotropy_scores: Isotropy feature values
        weights: Dictionary of feature weights
        
    Returns:
        integrated_scores: Combined scores
    """
    if weights is None:
        weights = {"log_response": 0.4, "ringness": 0.4, "isotropy": 0.2}
    
    if len(log_responses) == 0:
        return np.empty(0)
    
    # Z-score normalization within this batch
    def zscore(x):
        if len(x) <= 1:
            return np.zeros_like(x)
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x > 0:
            return (x - mean_x) / std_x
        else:
            return x - mean_x
    
    # Normalize features
    log_z = zscore(log_responses)
    ringness_z = zscore(-ringness_scores)  # Negative because we want negative ringness
    isotropy_z = zscore(isotropy_scores)
    
    # Compute weighted sum
    integrated_scores = (
        weights["log_response"] * log_z +
        weights["ringness"] * ringness_z +
        weights["isotropy"] * isotropy_z
    )
    
    return integrated_scores
