"""
Quality control and visualization functions.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def create_detection_summary(
    detections: pd.DataFrame,
    processing_time: float,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create summary statistics for detections.
    
    Args:
        detections: DataFrame with detection results
        processing_time: Total processing time in seconds
        config: Configuration dictionary
        
    Returns:
        Summary dictionary
    """
    if len(detections) == 0:
        return {
            "n_detections": 0,
            "processing_time_sec": processing_time,
            "detections_per_second": 0.0,
            "config": config
        }
    
    summary = {
        "n_detections": len(detections),
        "processing_time_sec": processing_time,
        "detections_per_second": len(detections) / processing_time if processing_time > 0 else 0,
        "radius_stats": {
            "mean_nm": float(detections["r_nm"].mean()),
            "std_nm": float(detections["r_nm"].std()),
            "min_nm": float(detections["r_nm"].min()),
            "max_nm": float(detections["r_nm"].max()),
            "median_nm": float(detections["r_nm"].median())
        },
        "score_stats": {
            "mean": float(detections["score"].mean()),
            "std": float(detections["score"].std()),
            "min": float(detections["score"].min()),
            "max": float(detections["score"].max()),
            "median": float(detections["score"].median())
        },
        "spatial_distribution": {
            "z_range": [float(detections["z_vox"].min()), float(detections["z_vox"].max())],
            "y_range": [float(detections["y_vox"].min()), float(detections["y_vox"].max())],
            "x_range": [float(detections["x_vox"].min()), float(detections["x_vox"].max())]
        },
        "config": config
    }
    
    return summary


def save_detection_cutouts(
    volume: np.ndarray,
    detections: pd.DataFrame,
    output_dir: Path,
    cutout_size: int = 32,
    max_cutouts: int = 100,
    voxel_size_nm: Tuple[float, float, float] = (8, 8, 8)
) -> List[str]:
    """
    Save cutout images around detected SVs.
    
    Args:
        volume: Input volume
        detections: Detection results
        output_dir: Output directory
        cutout_size: Size of cutouts in voxels
        max_cutouts: Maximum number of cutouts to save
        voxel_size_nm: Voxel size for scale bar
        
    Returns:
        List of saved cutout file paths
    """
    if not HAS_OPENCV:
        print("OpenCV not available, skipping cutout generation")
        return []
    
    cutout_dir = output_dir / "cutouts"
    cutout_dir.mkdir(exist_ok=True)
    
    saved_files = []
    n_cutouts = min(len(detections), max_cutouts)
    
    # Sort by score and take top detections
    top_detections = detections.nlargest(n_cutouts, 'score')
    
    half_size = cutout_size // 2
    
    for i, (_, detection) in enumerate(top_detections.iterrows()):
        z, y, x = int(detection['z_vox']), int(detection['y_vox']), int(detection['x_vox'])
        
        # Extract cutout
        z_start, z_end = max(0, z - half_size), min(volume.shape[0], z + half_size)
        y_start, y_end = max(0, y - half_size), min(volume.shape[1], y + half_size)
        x_start, x_end = max(0, x - half_size), min(volume.shape[2], x + half_size)
        
        cutout = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if cutout.size == 0:
            continue
        
        # Create MIP (Maximum Intensity Projection)
        mip_z = np.max(cutout, axis=0)
        mip_y = np.max(cutout, axis=1)
        mip_x = np.max(cutout, axis=2)
        
        # Normalize for display
        def normalize_for_display(img):
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return (img_norm * 255).astype(np.uint8)
        
        mip_z_norm = normalize_for_display(mip_z)
        mip_y_norm = normalize_for_display(mip_y)
        mip_x_norm = normalize_for_display(mip_x)
        
        # Create composite image
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(mip_z_norm, cmap='gray')
        axes[0].set_title(f'Z-projection\nScore: {detection["score"]:.3f}')
        axes[0].axis('off')
        
        axes[1].imshow(mip_y_norm, cmap='gray')
        axes[1].set_title(f'Y-projection\nRadius: {detection["r_nm"]:.1f} nm')
        axes[1].axis('off')
        
        axes[2].imshow(mip_x_norm, cmap='gray')
        axes[2].set_title(f'X-projection\nID: {detection["id"]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        cutout_file = cutout_dir / f"cutout_{i:03d}_score_{detection['score']:.3f}.png"
        plt.savefig(cutout_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_files.append(str(cutout_file))
    
    return saved_files


def create_qc_plots(
    detections: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any]
) -> List[str]:
    """
    Create quality control plots.
    
    Args:
        detections: Detection results
        output_dir: Output directory
        config: Configuration dictionary
        
    Returns:
        List of saved plot file paths
    """
    qc_dir = output_dir / "qc_plots"
    qc_dir.mkdir(exist_ok=True)
    
    saved_files = []
    
    if len(detections) == 0:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No detections found', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        empty_plot_file = qc_dir / "no_detections.png"
        plt.savefig(empty_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return [str(empty_plot_file)]
    
    # 1. Score distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(detections['score'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Detection Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Detection Scores')
    ax.grid(True, alpha=0.3)
    
    score_dist_file = qc_dir / "score_distribution.png"
    plt.savefig(score_dist_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(score_dist_file))
    
    # 2. Radius distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(detections['r_nm'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of SV Radii')
    ax.grid(True, alpha=0.3)
    
    # Add expected range
    expected_range = config.get('diameter_range_nm', [30, 50])
    expected_radii = [r/2 for r in expected_range]
    ax.axvspan(expected_radii[0], expected_radii[1], alpha=0.2, color='red', 
               label=f'Expected range: {expected_radii[0]:.1f}-{expected_radii[1]:.1f} nm')
    ax.legend()
    
    radius_dist_file = qc_dir / "radius_distribution.png"
    plt.savefig(radius_dist_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(radius_dist_file))
    
    # 3. Score vs Radius scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(detections['r_nm'], detections['score'], 
                        alpha=0.6, s=20, c=detections['score'], cmap='viridis')
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Detection Score')
    ax.set_title('Detection Score vs SV Radius')
    plt.colorbar(scatter, label='Score')
    ax.grid(True, alpha=0.3)
    
    score_vs_radius_file = qc_dir / "score_vs_radius.png"
    plt.savefig(score_vs_radius_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(score_vs_radius_file))
    
    # 4. Spatial distribution (if we have tile information)
    if 'tile_id' in detections.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 3D scatter plot projected to 2D
        scatter = ax.scatter(detections['x_vox'], detections['y_vox'], 
                           c=detections['z_vox'], s=20, alpha=0.6, cmap='plasma')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        ax.set_title('Spatial Distribution of Detections (colored by Z)')
        plt.colorbar(scatter, label='Z (voxels)')
        ax.grid(True, alpha=0.3)
        
        spatial_dist_file = qc_dir / "spatial_distribution.png"
        plt.savefig(spatial_dist_file, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(spatial_dist_file))
    
    # 5. Summary statistics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Detection count by tile (if available)
    if 'tile_id' in detections.columns:
        tile_counts = detections['tile_id'].value_counts().sort_index()
        ax1.bar(range(len(tile_counts)), tile_counts.values)
        ax1.set_xlabel('Tile ID')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detections per Tile')
    else:
        ax1.text(0.5, 0.5, 'No tile information', ha='center', va='center')
        ax1.set_title('Detections per Tile')
    
    # Score percentiles
    score_percentiles = np.percentile(detections['score'], [10, 25, 50, 75, 90])
    ax2.bar(range(len(score_percentiles)), score_percentiles, 
           tick_label=['10%', '25%', '50%', '75%', '90%'])
    ax2.set_ylabel('Score')
    ax2.set_title('Score Percentiles')
    
    # Radius vs depth (Z)
    ax3.scatter(detections['z_vox'], detections['r_nm'], alpha=0.5, s=10)
    ax3.set_xlabel('Z (voxels)')
    ax3.set_ylabel('Radius (nm)')
    ax3.set_title('Radius vs Depth')
    
    # Detection density
    if len(detections) > 100:
        ax4.hist2d(detections['x_vox'], detections['y_vox'], bins=20, cmap='Blues')
        ax4.set_xlabel('X (voxels)')
        ax4.set_ylabel('Y (voxels)')
        ax4.set_title('Detection Density (XY)')
    else:
        ax4.scatter(detections['x_vox'], detections['y_vox'], alpha=0.6, s=20)
        ax4.set_xlabel('X (voxels)')
        ax4.set_ylabel('Y (voxels)')  
        ax4.set_title('Detection Locations (XY)')
    
    plt.tight_layout()
    
    summary_plots_file = qc_dir / "summary_statistics.png"
    plt.savefig(summary_plots_file, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(str(summary_plots_file))
    
    return saved_files
