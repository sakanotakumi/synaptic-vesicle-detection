"""
Configuration classes for SV detection.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import yaml
import numpy as np


@dataclass
class SVDetectionConfig:
    """Configuration for synaptic vesicle detection."""
    
    # Input parameters
    voxel_size_nm: Tuple[float, float, float] = (8.0, 8.0, 8.0)  # z, y, x
    diameter_range_nm: Tuple[float, float] = (30.0, 50.0)
    
    # Tile processing
    tile_size: int = 256
    halo_size: int = 20
    
    # Preprocessing
    intensity_percentiles: Tuple[float, float] = (2.0, 98.0)
    normalization_method: str = "z_score"  # "z_score" or "min_max"
    
    # LoG/DoG parameters
    log_sigma_steps: int = 5
    scale_normalization: bool = True
    use_dog: bool = False  # Use DoG instead of LoG
    
    # Peak detection
    peak_threshold_percentile: float = 99.5
    min_distance_voxel: int = 3
    
    # NMS parameters
    nms_radius_factor: float = 0.9
    nms_min_radius: int = 3
    nms_max_radius: int = 10
    
    # Feature calculation
    ringness_inner_offset: int = 2
    ringness_outer_thickness: int = 2
    ringness_robustness_steps: int = 3
    
    # Sphericity (Hessian)
    isotropy_threshold: float = 0.7
    min_hessian_eigenvalue: float = 0.01
    hessian_sigma_factor: float = 1.0
    
    # Scoring weights
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "log_response": 0.4,
        "ringness": 0.4,
        "isotropy": 0.2
    })
    
    # Final filtering
    score_threshold: Optional[float] = None  # Auto-determined if None
    max_detections_per_volume_voxel: float = 100.0  # per 10^6 voxels
    
    # Performance
    n_workers: int = -1  # -1 for all cores
    use_gpu: bool = False
    chunk_size_mb: int = 512
    
    # Random seed
    random_seed: int = 47
    
    # Output
    output_formats: List[str] = field(default_factory=lambda: ["parquet", "napari_csv"])
    save_qc_images: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SVDetectionConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def get_radius_range_voxel(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert diameter range to radius range in voxels."""
        r_min_nm, r_max_nm = np.array(self.diameter_range_nm) / 2.0
        
        # Convert to voxels (for each axis)
        r_min_vox = r_min_nm / np.array(self.voxel_size_nm)
        r_max_vox = r_max_nm / np.array(self.voxel_size_nm)
        
        return r_min_vox, r_max_vox
    
    def get_sigma_list(self) -> np.ndarray:
        """Generate sigma values for LoG/DoG based on expected SV sizes."""
        r_min_vox, r_max_vox = self.get_radius_range_voxel()
        
        # Use average voxel size for isotropic sigma calculation
        avg_voxel_size = np.mean(self.voxel_size_nm)
        r_min_avg = (self.diameter_range_nm[0] / 2.0) / avg_voxel_size
        r_max_avg = (self.diameter_range_nm[1] / 2.0) / avg_voxel_size
        
        # Convert radius to sigma: r ≈ √3 * σ
        sigma_min = r_min_avg / np.sqrt(3)
        sigma_max = r_max_avg / np.sqrt(3)
        
        return np.linspace(sigma_min, sigma_max, self.log_sigma_steps)
    
    def get_nms_radius(self, estimated_radius_voxel: float) -> int:
        """Calculate NMS radius based on estimated SV radius."""
        nms_radius = int(round(self.nms_radius_factor * estimated_radius_voxel))
        return np.clip(nms_radius, self.nms_min_radius, self.nms_max_radius)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.diameter_range_nm[0] >= self.diameter_range_nm[1]:
            raise ValueError("diameter_range_nm[0] must be less than diameter_range_nm[1]")
        
        if any(v <= 0 for v in self.voxel_size_nm):
            raise ValueError("All voxel sizes must be positive")
        
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        
        if self.halo_size < 0:
            raise ValueError("halo_size must be non-negative")
        
        if not (0 <= self.peak_threshold_percentile <= 100):
            raise ValueError("peak_threshold_percentile must be between 0 and 100")
        
        if not (0 <= self.isotropy_threshold <= 1):
            raise ValueError("isotropy_threshold must be between 0 and 1")
        
        if sum(self.score_weights.values()) <= 0:
            raise ValueError("Sum of score weights must be positive")


def create_default_config() -> SVDetectionConfig:
    """Create a default configuration."""
    return SVDetectionConfig()


def load_config(config_path: str) -> SVDetectionConfig:
    """Load configuration from file."""
    return SVDetectionConfig.from_yaml(config_path)
