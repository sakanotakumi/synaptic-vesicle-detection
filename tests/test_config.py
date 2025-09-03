"""
Tests for the SV detection configuration.
"""

import pytest
import numpy as np
from sv_detection.config import SVDetectionConfig, create_default_config


def test_default_config():
    """Test that default configuration is valid."""
    config = create_default_config()
    config.validate()  # Should not raise
    
    assert config.voxel_size_nm == (8.0, 8.0, 8.0)
    assert config.diameter_range_nm == (30.0, 50.0)
    assert config.tile_size == 256
    assert config.random_seed == 47


def test_config_validation():
    """Test configuration validation."""
    config = create_default_config()
    
    # Valid config should pass
    config.validate()
    
    # Invalid diameter range
    config.diameter_range_nm = (50.0, 30.0)  # max < min
    with pytest.raises(ValueError):
        config.validate()
    
    # Negative voxel size
    config = create_default_config()
    config.voxel_size_nm = (-1.0, 8.0, 8.0)
    with pytest.raises(ValueError):
        config.validate()
    
    # Invalid tile size
    config = create_default_config()
    config.tile_size = -1
    with pytest.raises(ValueError):
        config.validate()


def test_radius_conversion():
    """Test radius range conversion."""
    config = create_default_config()
    config.voxel_size_nm = (8.0, 8.0, 8.0)
    config.diameter_range_nm = (32.0, 48.0)
    
    r_min_vox, r_max_vox = config.get_radius_range_voxel()
    
    # Expected radii in voxels: diameter/2 / voxel_size
    expected_r_min = np.array([16.0/8.0, 16.0/8.0, 16.0/8.0])  # [2, 2, 2]
    expected_r_max = np.array([24.0/8.0, 24.0/8.0, 24.0/8.0])  # [3, 3, 3]
    
    np.testing.assert_array_equal(r_min_vox, expected_r_min)
    np.testing.assert_array_equal(r_max_vox, expected_r_max)


def test_sigma_list_generation():
    """Test sigma list generation."""
    config = create_default_config()
    config.diameter_range_nm = (24.0, 48.0)  # 12-24 nm radius
    config.voxel_size_nm = (8.0, 8.0, 8.0)
    config.log_sigma_steps = 3
    
    sigma_list = config.get_sigma_list()
    
    assert len(sigma_list) == 3
    assert sigma_list[0] < sigma_list[-1]  # Should be increasing
    
    # Check that sigma values are reasonable
    # radius = sigma * sqrt(3), so sigma = radius / sqrt(3)
    # Expected radius range in voxels: 1.5 - 3.0
    # Expected sigma range: ~0.87 - 1.73
    assert 0.5 < sigma_list[0] < 1.5
    assert 1.0 < sigma_list[-1] < 2.5


def test_nms_radius_calculation():
    """Test NMS radius calculation."""
    config = create_default_config()
    
    # Test with various radius values
    test_radii = [1.0, 2.5, 5.0, 10.0, 20.0]
    
    for radius in test_radii:
        nms_radius = config.get_nms_radius(radius)
        
        # Should be clipped to valid range
        assert config.nms_min_radius <= nms_radius <= config.nms_max_radius
        
        # Should be roughly proportional to input radius (when not clipped)
        if 3 <= radius * config.nms_radius_factor <= 10:
            expected = int(round(radius * config.nms_radius_factor))
            assert nms_radius == expected


def test_config_serialization(tmp_path):
    """Test YAML serialization and deserialization."""
    config = create_default_config()
    config.voxel_size_nm = (4.0, 4.0, 4.0)
    config.diameter_range_nm = (20.0, 40.0)
    
    # Save to file
    yaml_path = tmp_path / "test_config.yaml"
    config.to_yaml(str(yaml_path))
    
    # Load from file
    loaded_config = SVDetectionConfig.from_yaml(str(yaml_path))
    
    # Check that values match
    assert loaded_config.voxel_size_nm == (4.0, 4.0, 4.0)
    assert loaded_config.diameter_range_nm == (20.0, 40.0)
    assert loaded_config.tile_size == config.tile_size
    assert loaded_config.random_seed == config.random_seed
