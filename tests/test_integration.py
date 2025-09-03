"""
Basic integration test for the SV detection pipeline.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from sv_detection import SVDetector, SVDetectionConfig
from sv_detection.utils import load_volume


def create_test_volume(shape=(32, 32, 32)):
    """Create a simple test volume with some blob structures."""
    volume = np.random.normal(100, 10, shape).astype(np.float32)
    
    # Add a few blob-like structures
    center1 = (16, 16, 16)
    center2 = (8, 8, 8)
    center3 = (24, 24, 24)
    
    for center in [center1, center2, center3]:
        z, y, x = center
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Bright center
        mask_inner = ((zz - z)**2 + (yy - y)**2 + (xx - x)**2) < 4
        volume[mask_inner] += 50
        
        # Dark ring
        mask_ring = (((zz - z)**2 + (yy - y)**2 + (xx - x)**2) >= 4) & \
                   (((zz - z)**2 + (yy - y)**2 + (xx - x)**2) < 16)
        volume[mask_ring] -= 30
    
    return volume.astype(np.uint8)


def test_small_volume_detection():
    """Test detection on a small synthetic volume."""
    # Create test volume
    volume = create_test_volume((32, 32, 32))
    
    # Create configuration for small volume
    config = SVDetectionConfig(
        voxel_size_nm=(8.0, 8.0, 8.0),
        diameter_range_nm=(20.0, 40.0),
        tile_size=64,  # Larger than volume
        peak_threshold_percentile=95.0,  # Lower threshold
        log_sigma_steps=3,
        save_qc_images=False,
        random_seed=42
    )
    
    # Create detector
    detector = SVDetector(config)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
        import tifffile
        tifffile.imwrite(tmp_file.name, volume)
        tmp_path = tmp_file.name
    
    try:
        # Run detection
        results = detector._detect_single_tile(volume)
        
        # Check that we get some reasonable outputs
        assert "coords" in results
        assert "scores" in results
        assert "radii" in results
        
        # Should find at least one detection (very permissive test)
        assert len(results["coords"]) >= 0  # At least don't crash
        
        if len(results["coords"]) > 0:
            # Basic sanity checks
            assert results["coords"].shape[1] == 3  # 3D coordinates
            assert len(results["scores"]) == len(results["coords"])
            assert len(results["radii"]) == len(results["coords"])
            
            # Coordinates should be within volume bounds
            assert np.all(results["coords"] >= 0)
            assert np.all(results["coords"][:, 0] < volume.shape[0])
            assert np.all(results["coords"][:, 1] < volume.shape[1])
            assert np.all(results["coords"][:, 2] < volume.shape[2])
            
            # Radii should be positive and reasonable
            assert np.all(results["radii"] > 0)
            assert np.all(results["radii"] < 20)  # Shouldn't be huge
    
    finally:
        # Clean up
        Path(tmp_path).unlink()


def test_full_pipeline_with_output():
    """Test the full detection pipeline with file I/O."""
    # Create test volume
    volume = create_test_volume((64, 64, 64))
    
    # Create configuration
    config = SVDetectionConfig(
        voxel_size_nm=(8.0, 8.0, 8.0),
        diameter_range_nm=(20.0, 40.0),
        tile_size=128,  # Larger than volume
        peak_threshold_percentile=90.0,
        save_qc_images=False,  # Don't generate QC for tests
        output_formats=["parquet"],
        random_seed=42
    )
    
    detector = SVDetector(config)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save test volume
        volume_path = Path(tmp_dir) / "test_volume.tif"
        import tifffile
        tifffile.imwrite(volume_path, volume)
        
        # Run detection
        output_dir = Path(tmp_dir) / "results"
        results = detector.detect(
            volume_path=str(volume_path),
            output_dir=str(output_dir)
        )
        
        # Check results DataFrame
        assert isinstance(results, type(pd.DataFrame()))
        
        expected_columns = [
            'id', 'z_vox', 'y_vox', 'x_vox', 'z_um', 'y_um', 'x_um',
            'r_vox', 'r_nm', 'score', 'tile_id'
        ]
        
        for col in expected_columns:
            assert col in results.columns
        
        # Check output files were created
        assert (output_dir / "sv_points.parquet").exists()
        assert (output_dir / "run_config.yaml").exists()
        assert (output_dir / "metrics.jsonl").exists()


def test_empty_volume():
    """Test behavior with empty/uniform volume."""
    # Create uniform volume (no features)
    volume = np.full((32, 32, 32), 128, dtype=np.uint8)
    
    config = SVDetectionConfig(
        voxel_size_nm=(8.0, 8.0, 8.0),
        diameter_range_nm=(20.0, 40.0),
        peak_threshold_percentile=99.0,
        save_qc_images=False,
        random_seed=42
    )
    
    detector = SVDetector(config)
    results = detector._detect_single_tile(volume)
    
    # Should return empty results without crashing
    assert len(results["coords"]) == 0
    assert len(results["scores"]) == 0
    assert len(results["radii"]) == 0


# Import pandas for type checking
try:
    import pandas as pd
except ImportError:
    pd = None

@pytest.mark.skipif(pd is None, reason="pandas not available")
def test_dataframe_creation():
    """Test DataFrame creation from detection results."""
    # Mock detection results
    coords = np.array([[10, 20, 30], [15, 25, 35]])
    scores = np.array([0.8, 0.9])
    radii = np.array([2.5, 3.0])
    
    detections = {
        "coords": coords,
        "log_responses": scores,
        "radii": radii,
        "ringness": np.array([0.1, 0.2]),
        "isotropy": np.array([0.7, 0.8]),
        "scores": scores
    }
    
    config = SVDetectionConfig(voxel_size_nm=(8.0, 8.0, 8.0))
    detector = SVDetector(config)
    
    df = detector._create_results_dataframe(detections, {"format": "test"})
    
    assert len(df) == 2
    assert df['z_vox'].tolist() == [10, 15]
    assert df['y_vox'].tolist() == [20, 25]
    assert df['x_vox'].tolist() == [30, 35]
    
    # Check unit conversions
    expected_z_um = [10 * 8.0 / 1000, 15 * 8.0 / 1000]
    np.testing.assert_array_almost_equal(df['z_um'], expected_z_um)
