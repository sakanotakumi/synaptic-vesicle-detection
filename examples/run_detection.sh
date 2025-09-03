#!/bin/bash

# Example shell script for SV detection using command line interface

# Set up paths
VOLUME_PATH="path/to/your/volume.tif"
MASK_PATH="path/to/your/mask.tif"  # Optional
OUTPUT_DIR="results/sv_detection_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="examples/default_config.yaml"

# Check if volume exists
if [ ! -f "$VOLUME_PATH" ]; then
    echo "Error: Volume file not found: $VOLUME_PATH"
    echo "Please update VOLUME_PATH in this script"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== SV Detection ==="
echo "Volume: $VOLUME_PATH"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"

# Run detection with default parameters
sv-detect "$VOLUME_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --voxel-size 8.0 8.0 8.0 \
    --diam-range 30.0 50.0 \
    --tile-size 256 \
    --halo-size 20 \
    --peak-threshold 99.5 \
    --max-per-million 100.0 \
    --output-formats parquet napari_csv \
    --save-qc \
    --verbose

# Check if detection was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Detection completed successfully! ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    ls -la "$OUTPUT_DIR"
    
    # Show sample of results if parquet file exists
    if [ -f "$OUTPUT_DIR/sv_points.parquet" ]; then
        echo ""
        echo "Sample results (first 10 detections):"
        python -c "
import pandas as pd
df = pd.read_parquet('$OUTPUT_DIR/sv_points.parquet')
print(f'Total detections: {len(df)}')
if len(df) > 0:
    print(df.head(10)[['id', 'z_vox', 'y_vox', 'x_vox', 'r_nm', 'score']])
"
    fi
else
    echo "Detection failed!"
    exit 1
fi
