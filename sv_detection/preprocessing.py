"""
Core preprocessing functions for SV detection.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage
from skimage.filters import gaussian
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def normalize_intensity_tile(
    volume: np.ndarray,
    percentiles: Tuple[float, float] = (2.0, 98.0),
    method: str = "z_score"
) -> np.ndarray:
    """
    Normalize intensity for a single tile.
    
    Args:
        volume: Input volume tile
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


def apply_mask_to_volume(
    volume: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply binary mask to volume, setting masked regions to zero.
    
    Args:
        volume: Input volume
        mask: Binary mask (True for valid regions)
        
    Returns:
        Masked volume
    """
    if mask is None:
        return volume
    
    if mask.shape != volume.shape:
        raise ValueError(f"Mask shape {mask.shape} doesn't match volume shape {volume.shape}")
    
    volume_masked = volume.copy()
    volume_masked[~mask] = 0
    return volume_masked


def gaussian_filter_3d(
    volume: np.ndarray, 
    sigma: float, 
    use_gpu: bool = False
) -> np.ndarray:
    """
    Apply 3D Gaussian filter.
    
    Args:
        volume: Input volume
        sigma: Gaussian sigma
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Filtered volume
    """
    if use_gpu and HAS_CUPY:
        volume_gpu = cp.asarray(volume)
        # CuPy doesn't have gaussian filter, use convolution
        from cupyx.scipy import ndimage as cupy_ndimage
        filtered_gpu = cupy_ndimage.gaussian_filter(volume_gpu, sigma)
        return cp.asnumpy(filtered_gpu)
    else:
        return gaussian(volume, sigma=sigma, preserve_range=True)


def laplacian_of_gaussian_3d(
    volume: np.ndarray, 
    sigma: float, 
    use_gpu: bool = False,
    scale_normalize: bool = True
) -> np.ndarray:
    """
    Compute scale-normalized Laplacian of Gaussian.
    
    Args:
        volume: Input volume
        sigma: Gaussian sigma
        use_gpu: Whether to use GPU acceleration
        scale_normalize: Whether to apply scale normalization
        
    Returns:
        LoG response
    """
    if use_gpu and HAS_CUPY:
        volume_gpu = cp.asarray(volume)
        from cupyx.scipy import ndimage as cupy_ndimage
        
        # Apply Gaussian filter
        gaussian_filtered = cupy_ndimage.gaussian_filter(volume_gpu, sigma)
        
        # Apply Laplacian
        laplacian_kernel = cp.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                   [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=cp.float32)
        
        log_response = cupy_ndimage.convolve(gaussian_filtered, laplacian_kernel)
        
        if scale_normalize:
            log_response *= sigma**2
        
        # Make positive for dark spots with bright centers
        log_response = -log_response
        
        return cp.asnumpy(log_response)
    else:
        # Use scipy
        gaussian_filtered = ndimage.gaussian_filter(volume, sigma)
        log_response = ndimage.laplace(gaussian_filtered)
        
        if scale_normalize:
            log_response *= sigma**2
        
        # Make positive for dark spots with bright centers  
        log_response = -log_response
        
        return log_response


def difference_of_gaussians_3d(
    volume: np.ndarray,
    sigma1: float,
    sigma2: float,
    use_gpu: bool = False,
    scale_normalize: bool = True
) -> np.ndarray:
    """
    Compute Difference of Gaussians (DoG).
    
    Args:
        volume: Input volume
        sigma1: First Gaussian sigma (smaller)
        sigma2: Second Gaussian sigma (larger)
        use_gpu: Whether to use GPU acceleration
        scale_normalize: Whether to apply scale normalization
        
    Returns:
        DoG response
    """
    if sigma1 >= sigma2:
        sigma1, sigma2 = sigma2, sigma1
    
    if use_gpu and HAS_CUPY:
        volume_gpu = cp.asarray(volume)
        from cupyx.scipy import ndimage as cupy_ndimage
        
        gaussian1 = cupy_ndimage.gaussian_filter(volume_gpu, sigma1)
        gaussian2 = cupy_ndimage.gaussian_filter(volume_gpu, sigma2)
        
        dog_response = gaussian1 - gaussian2
        
        if scale_normalize:
            # Scale normalization for DoG
            dog_response *= (sigma2**2 - sigma1**2)
        
        return cp.asnumpy(dog_response)
    else:
        gaussian1 = ndimage.gaussian_filter(volume, sigma1)
        gaussian2 = ndimage.gaussian_filter(volume, sigma2)
        
        dog_response = gaussian1 - gaussian2
        
        if scale_normalize:
            # Scale normalization for DoG
            dog_response *= (sigma2**2 - sigma1**2)
        
        return dog_response


def multiscale_response(
    volume: np.ndarray,
    sigma_list: np.ndarray,
    use_dog: bool = False,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multiscale LoG or DoG response.
    
    Args:
        volume: Input volume
        sigma_list: Array of sigma values
        use_dog: Whether to use DoG instead of LoG
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        max_response: Maximum response across scales
        best_scale_idx: Index of best scale for each voxel
    """
    responses = []
    
    if use_dog:
        # For DoG, we need pairs of sigmas
        for i in range(len(sigma_list) - 1):
            sigma1, sigma2 = sigma_list[i], sigma_list[i + 1]
            response = difference_of_gaussians_3d(
                volume, sigma1, sigma2, use_gpu=use_gpu
            )
            responses.append(response)
    else:
        # For LoG
        for sigma in sigma_list:
            response = laplacian_of_gaussian_3d(
                volume, sigma, use_gpu=use_gpu
            )
            responses.append(response)
    
    # Stack responses
    response_stack = np.stack(responses, axis=0)
    
    # Find maximum response and corresponding scale
    max_response = np.max(response_stack, axis=0)
    best_scale_idx = np.argmax(response_stack, axis=0)
    
    return max_response, best_scale_idx
