"""
Example script for SV detection using the Python API.
"""

import numpy as np
from pathlib import Path
from sv_detection import SVDetector, SVDetectionConfig


def main():
    """Example of using SV detection programmatically."""
    
    # Create configuration
    config = SVDetectionConfig(
        voxel_size_nm=(8.0, 8.0, 8.0),
        diameter_range_nm=(30.0, 50.0),
        tile_size=256,
        use_gpu=False,
        save_qc_images=True,
        output_formats=["parquet", "napari_csv"]
    )
    
    # Print configuration
    print("Configuration:")
    print(f"  Voxel size: {config.voxel_size_nm} nm")
    print(f"  Expected SV diameter: {config.diameter_range_nm} nm")
    print(f"  Tile size: {config.tile_size}")
    print(f"  Using GPU: {config.use_gpu}")
    
    # Create detector
    detector = SVDetector(config)
    
    # Run detection (replace with your actual file paths)
    volume_path = "path/to/your/volume.tif"
    mask_path = None  # Optional: "path/to/your/mask.tif"
    output_dir = "results/"
    
    try:
        # Check if volume exists
        if not Path(volume_path).exists():
            print(f"Creating synthetic test volume for demonstration...")
            create_synthetic_volume(volume_path)
        
        print(f"Running detection on {volume_path}")
        results = detector.detect(
            volume_path=volume_path,
            mask_path=mask_path,
            output_dir=output_dir
        )
        
        print(f"\nResults:")
        print(f"  Found {len(results)} SVs")
        
        if len(results) > 0:
            print(f"  Average radius: {results['r_nm'].mean():.1f} nm")
            print(f"  Score range: {results['score'].min():.3f} - {results['score'].max():.3f}")
            print(f"  Spatial extent:")
            print(f"    Z: {results['z_vox'].min():.0f} - {results['z_vox'].max():.0f}")
            print(f"    Y: {results['y_vox'].min():.0f} - {results['y_vox'].max():.0f}")
            print(f"    X: {results['x_vox'].min():.0f} - {results['x_vox'].max():.0f}")
        
        print(f"\nResults saved to: {output_dir}")
        
    except FileNotFoundError:
        print(f"Volume file not found: {volume_path}")
        print("Please update the volume_path variable with your actual file path")
    except Exception as e:
        print(f"Error during detection: {e}")


def create_synthetic_volume(output_path, size=(64, 64, 64)):
    """
    Create a synthetic test volume with some blob-like structures.
    This is just for testing - replace with your actual data.
    """
    print(f"Creating synthetic volume: {size}")
    
    # Create base volume with noise
    volume = np.random.normal(100, 20, size).astype(np.uint16)
    
    # Add some blob-like structures
    from scipy.ndimage import gaussian_filter
    
    # Create some random blob centers
    n_blobs = 20
    for i in range(n_blobs):
        z = np.random.randint(10, size[0] - 10)
        y = np.random.randint(10, size[1] - 10)
        x = np.random.randint(10, size[2] - 10)
        
        # Create blob
        zz, yy, xx = np.ogrid[:size[0], :size[1], :size[2]]
        blob_mask = ((zz - z)**2 + (yy - y)**2 + (xx - x)**2) < (3 + np.random.rand() * 2)**2
        
        # Add bright center and dark surround
        volume[blob_mask] += 50
        
        # Dark surround
        surround_mask = ((zz - z)**2 + (yy - y)**2 + (xx - x)**2) < (5 + np.random.rand() * 3)**2
        surround_mask = surround_mask & ~blob_mask
        volume[surround_mask] -= 30
    
    # Smooth slightly
    volume = gaussian_filter(volume.astype(float), sigma=0.5)
    volume = np.clip(volume, 0, 255).astype(np.uint8)
    
    # Save as TIFF
    from tifffile import imwrite
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path, volume)
    print(f"Synthetic volume saved to: {output_path}")


if __name__ == "__main__":
    main()
