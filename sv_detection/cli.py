"""
Command-line interface for SV detection.
"""

import click
from pathlib import Path
import time
import sys

from .config import SVDetectionConfig, create_default_config
from .detector import SVDetector
from . import __version__


@click.command()
@click.argument('volume_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), required=True,
              help='Output directory for results')
@click.option('--mask-path', '-m', type=click.Path(exists=True), default=None,
              help='Optional binary mask file')
@click.option('--config', '-c', type=click.Path(exists=True), default=None,
              help='Configuration YAML file')
@click.option('--voxel-size', nargs=3, type=float, default=(8.0, 8.0, 8.0),
              help='Voxel size in nm (z y x)')
@click.option('--diam-range', nargs=2, type=float, default=(30.0, 50.0),
              help='Expected SV diameter range in nm (min max)')
@click.option('--tile-size', type=int, default=256,
              help='Tile size for processing large volumes')
@click.option('--halo-size', type=int, default=20,
              help='Halo size for tile overlap')
@click.option('--use-gpu/--no-gpu', default=False,
              help='Use GPU acceleration if available')
@click.option('--n-workers', type=int, default=-1,
              help='Number of parallel workers (-1 for all cores)')
@click.option('--peak-threshold', type=float, default=99.5,
              help='Percentile threshold for peak detection')
@click.option('--score-threshold', type=float, default=None,
              help='Minimum score threshold for final detections')
@click.option('--max-per-million', type=float, default=100.0,
              help='Maximum detections per million voxels')
@click.option('--output-formats', multiple=True, 
              default=['parquet', 'napari_csv'],
              help='Output formats (parquet, csv, napari_csv)')
@click.option('--save-qc/--no-qc', default=True,
              help='Save quality control images and plots')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def main(volume_path, output_dir, mask_path, config, voxel_size, diam_range,
         tile_size, halo_size, use_gpu, n_workers, peak_threshold, 
         score_threshold, max_per_million, output_formats, save_qc, verbose):
    """
    Automatic synaptic vesicle detection from FIB-SEM 3D volumes.
    
    VOLUME_PATH: Path to input volume (TIFF stack, Zarr, etc.)
    """
    
    if verbose:
        print(f"SV Detection v{__version__}")
        print(f"Input volume: {volume_path}")
        print(f"Output directory: {output_dir}")
        if mask_path:
            print(f"Mask: {mask_path}")
    
    try:
        # Load or create configuration
        if config:
            if verbose:
                print(f"Loading configuration from {config}")
            detection_config = SVDetectionConfig.from_yaml(config)
        else:
            if verbose:
                print("Using default configuration with CLI overrides")
            detection_config = create_default_config()
        
        # Override config with CLI arguments
        detection_config.voxel_size_nm = voxel_size
        detection_config.diameter_range_nm = diam_range
        detection_config.tile_size = tile_size
        detection_config.halo_size = halo_size
        detection_config.use_gpu = use_gpu
        detection_config.n_workers = n_workers
        detection_config.peak_threshold_percentile = peak_threshold
        detection_config.score_threshold = score_threshold
        detection_config.max_detections_per_volume_voxel = max_per_million
        detection_config.output_formats = list(output_formats)
        detection_config.save_qc_images = save_qc
        
        # Validate configuration
        detection_config.validate()
        
        if verbose:
            print(f"Voxel size: {detection_config.voxel_size_nm} nm")
            print(f"Expected SV diameter: {detection_config.diameter_range_nm} nm")
            print(f"Tile size: {detection_config.tile_size}")
            print(f"Using GPU: {detection_config.use_gpu}")
            print(f"Workers: {detection_config.n_workers}")
        
        # Create detector
        detector = SVDetector(detection_config)
        
        # Run detection
        start_time = time.time()
        results = detector.detect(
            volume_path=volume_path,
            mask_path=mask_path,
            output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Detection completed successfully!")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Detections found: {len(results)}")
        
        if len(results) > 0:
            print(f"Average radius: {results['r_nm'].mean():.1f} ± {results['r_nm'].std():.1f} nm")
            print(f"Score range: {results['score'].min():.3f} - {results['score'].max():.3f}")
        
        print(f"Results saved to: {output_dir}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.argument('output_path', type=click.Path())
@click.option('--voxel-size', nargs=3, type=float, default=(8.0, 8.0, 8.0),
              help='Voxel size in nm (z y x)')
@click.option('--diam-range', nargs=2, type=float, default=(30.0, 50.0),
              help='Expected SV diameter range in nm (min max)')
@click.option('--tile-size', type=int, default=256,
              help='Tile size for processing')
def create_config(output_path, voxel_size, diam_range, tile_size):
    """Create a default configuration file."""
    
    config = create_default_config()
    config.voxel_size_nm = voxel_size
    config.diameter_range_nm = diam_range
    config.tile_size = tile_size
    
    config.to_yaml(output_path)
    print(f"Default configuration saved to: {output_path}")


@click.group()
def cli():
    """SV Detection CLI."""
    pass


cli.add_command(main, name='detect')
cli.add_command(create_config, name='create-config')


if __name__ == '__main__':
    cli()
