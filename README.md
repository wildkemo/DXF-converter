# Image Contour Detection and Vectorization Engine

A Python-based system for extracting geometric vector contours from 2D raster illustrations, diagrams, and line drawings.

## Features

- **Robust Preprocessing**: Handles RGBA transparency, applies noise-reduction filters, and performs CLAHE contrast enhancement.
- **Smart Binarization**: Auto-detects dark/light polarity and supports Otsu, Adaptive, and Canny methods.
- **Hierarchical Detection**: Accurately maps nested shapes and holes into a parent-child tree.
- **Geometry Simplification**: Ramer-Douglas-Peucker (RDP) algorithm with intelligent collinear vertex removal.
- **Contour Smoothing & Fairing**: Gaussian filtering along contours eliminates discrete pixel-staircase zigzags on arcs and curves while preserving true sharp corners.
- **CNC Offset Line Duplication**: Generates parallel offset toolpaths with custom offset distance for CNC cutter compensation and dual-pass machining. Duplicates are organized on dedicated CAD layers (`CNC_OFFSET`) and styled distinctly in SVG.
- **Ultra-Detailed Edge Detection**: Multi-channel color-space fusion (BGR + Lab) and hybrid edge detection capturing maximum contours and micro-details.
- **Winding Normalization**: Consistent CCW/CW orientation for CAD downstream compatibility.
- **JSON Serialization & DXF/SVG Export**: Structured JSON export, smooth DXF splines with hierarchy layers and extended XDATA, and layered SVG vector drawings.
- **CLI & Python API**: Usable as a command-line tool or integrated as a Python module.

## Installation

```bash
pip install -r requirements.txt
```

## CLI Usage

```bash
# Basic usage
python -m src.main --input input_image.png --output output_dir/

# CNC Toolpath mode with 3.0 unit line offset
python -m src.main --input image.jpg --output output_dir/ -D 3.0

# Maximum contour detection for complex drawings
python -m src.main --input image.jpg --output output_dir/ --detect-more

# CNC offset with dual passes (both sides) and hybrid edge detection
python -m src.main --input image.jpg --output output_dir/ -D 2.5 --duplicate-both-sides --threshold-method hybrid_all
```

Options:
- `-D, --duplicate-distance`: CNC duplicate line offset distance (in coordinate units/pixels, e.g. `3.0` or `-2.0`). Positive offsets outward; negative offsets inward.
- `--duplicate-both-sides`: Generate CNC offset lines on both sides (`+distance` and `-distance`).
- `--miter-limit`: Miter limit ratio for corner mitering during line offsetting (default: `2.5`).
- `--detect-more`: Ultra-detailed detection mode maximizing detected contours using multi-channel hybrid edges and micro-feature thresholds.
- `-t, --threshold-method`: `otsu`, `adaptive_gaussian`, `adaptive_mean`, `canny`, `binary_fixed`, `multi_channel_canny`, `hybrid_all`, `morphological_gradient`.
- `-c, --coord-space`: `pixel`, `normalized`, `cartesian_pixel`, `cartesian_normalized`.
- `-e, --epsilon-factor`: Simplification aggressiveness (default: `0.005`). Set `0.0` for full resolution curves.
- `--min-area`: Minimum contour area to keep (default: `10.0`).
- `--min-perimeter`: Minimum contour perimeter to keep (default: `10.0`).
- `--smooth / --no-smooth`: Enable/disable contour smoothing (default: enabled).
- `--smooth-sigma`: Smoothing Gaussian standard deviation (default: `1.5`).
- `--preserve-corners / --no-preserve-corners`: Keep sharp corners intact during curve smoothing (default: enabled).
- `--no-debug`: Disable output of debug visualization PNGs.

## Python API

```python
from src import VectorizationPipeline, VectorizationConfig, ThresholdMethod

# Configure pipeline with CNC offset duplication and enhanced detection
config = VectorizationConfig(
    threshold_method=ThresholdMethod.HYBRID_ALL,
    use_grayscale=True,
    duplicate_distance=3.0,
    duplicate_both_sides=False,
    smooth_contours=True
)
pipeline = VectorizationPipeline(config)

result = pipeline.process_image("path/to/image.png", output_dir="output/")
print(f"Extracted {result.metrics.total_contours} contours (including CNC offset toolpaths).")
```
