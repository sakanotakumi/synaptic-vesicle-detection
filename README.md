# Synaptic Vesicle Detection

Automatic detection of synaptic vesicles (SVs) from FIB-SEM 3D volumes using advanced image analysis techniques.

## Features

- **Multi-scale blob detection** using Laplacian of Gaussian (LoG) or Difference of Gaussians (DoG)
- **Advanced feature extraction** including ringness and sphericity measures
- **Non-maximum suppression** to eliminate duplicate detections
- **Scalable processing** with tile-based approach for large volumes
- **GPU acceleration** support (optional)
- **Quality control** outputs with visualization and metrics
- **Multiple output formats** (Parquet, CSV, Napari-compatible)

## Installation

### From source

```bash
git clone https://github.com/neurobiology-ut/synaptic-vesicle-detection.git
cd synaptic-vesicle-detection
pip install -e .
```

### With GPU support

```bash
pip install -e .[gpu]
```

## Quick Start

### Command Line Interface

```bash
# Basic usage
sv-detect volume.tif --output-dir results/

# With custom parameters
sv-detect volume.tif \
    --output-dir results/ \
    --voxel-size 8.0 8.0 8.0 \
    --diam-range 30.0 50.0 \
    --tile-size 256 \
    --use-gpu \
    --verbose

# With mask and configuration file
sv-detect volume.tif \
    --output-dir results/ \
    --mask-path mask.tif \
    --config my_config.yaml
```

### Python API

```python
from sv_detection import SVDetector, SVDetectionConfig

# Create configuration
config = SVDetectionConfig(
    voxel_size_nm=(8.0, 8.0, 8.0),
    diameter_range_nm=(30.0, 50.0),
    use_gpu=True
)

# Create detector and run
detector = SVDetector(config)
results = detector.detect("volume.tif", output_dir="results/")

print(f"Found {len(results)} synaptic vesicles")
```

## Configuration

The detection behavior can be customized through configuration files or API parameters:

### Key Parameters

- **`voxel_size_nm`**: Physical voxel size in nanometers (z, y, x)
- **`diameter_range_nm`**: Expected SV diameter range in nanometers
- **`tile_size`**: Tile size for processing large volumes
- **`peak_threshold_percentile`**: Percentile threshold for initial peak detection
- **`score_weights`**: Weights for combining different feature scores

### Example Configuration

```yaml
# Save as config.yaml
voxel_size_nm: [8.0, 8.0, 8.0]
diameter_range_nm: [30.0, 50.0]
tile_size: 256
peak_threshold_percentile: 99.5
score_weights:
  log_response: 0.4
  ringness: 0.4
  isotropy: 0.2
use_gpu: false
save_qc_images: true
```

## Input/Output

### Input

- **Volume**: TIFF stack, Zarr array, or other 3D image formats
- **Mask** (optional): Binary mask to restrict detection region
- **Configuration**: YAML file with detection parameters

### Output

- **`sv_points.parquet`**: Main results with columns:
  - `id`, `z_vox`, `y_vox`, `x_vox`: Detection ID and coordinates
  - `z_um`, `y_um`, `x_um`: Coordinates in micrometers
  - `r_vox`, `r_nm`: Radius in voxels and nanometers
  - `score`: Integrated detection score
  - `tile_id`: Source tile identifier

- **`sv_points_napari.csv`**: Napari Points layer compatible format
- **`metrics.jsonl`**: Processing statistics and performance metrics
- **`qc_plots/`**: Quality control plots and visualizations
- **`cutouts/`**: Sample detection cutouts for visual inspection

## Algorithm Overview

The detection pipeline consists of several stages:

1. **Preprocessing**: Intensity normalization and optional masking
2. **Multi-scale response**: Laplacian of Gaussian at multiple scales
3. **Peak detection**: Local maxima finding with threshold
4. **Feature extraction**:
   - **Ringness**: Contrast between center and surrounding ring
   - **Sphericity**: Hessian eigenvalue analysis for blob-like shapes
5. **Scoring**: Weighted combination of features
6. **Non-maximum suppression**: Elimination of overlapping detections
7. **Final filtering**: Score thresholding or top-K selection

## Performance

### Scalability

- **Small volumes** (< 500 MB): Single-chunk processing
- **Large volumes** (> 500 MB): Automatic tile-based processing
- **Multi-core**: Parallel tile processing
- **GPU acceleration**: Optional CUDA support for filtering operations

### Typical Performance

- **Processing speed**: ~1-10 million voxels/second (CPU)
- **Memory usage**: ~2-4x input volume size
- **Detection accuracy**: Depends on data quality and parameter tuning

## Examples

See the `examples/` directory for:

- `example_usage.py`: Python API usage
- `run_detection.sh`: Command-line usage
- `default_config.yaml`: Default configuration template

## Development

### Setup Development Environment

```bash
git clone https://github.com/neurobiology-ut/synaptic-vesicle-detection.git
cd synaptic-vesicle-detection
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black sv_detection/
flake8 sv_detection/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sv_detection,
  title={Automatic Synaptic Vesicle Detection},
  author={Neurobiology Lab},
  year={2024},
  url={https://github.com/neurobiology-ut/synaptic-vesicle-detection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions or issues, please open an issue on GitHub or contact the development team.

## Acknowledgments

This work was developed for automated analysis of FIB-SEM datasets in neurobiology research.
