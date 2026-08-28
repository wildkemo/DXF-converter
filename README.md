# Image Contour Detection and Vectorization Engine

A Python-based system for extracting geometric vector contours from 2D raster illustrations, diagrams, and line drawings.

## Features

- **Robust Preprocessing**: Handles RGBA transparency, applies noise-reduction filters, and performs CLAHE contrast enhancement.
- **Smart Binarization**: Auto-detects dark/light polarity and supports Otsu, Adaptive, and Canny methods.
- **Hierarchical Detection**: Accurately maps nested shapes and holes into a parent-child tree.
- **Geometry Simplification**: Ramer-Douglas-Peucker (RDP) algorithm with intelligent collinear vertex removal.
- **Winding Normalization**: Consistent CCW/CW orientation for CAD downstream compatibility.
- **JSON Serialization**: Structured JSON export with complete geometry and metadata.
- **CLI & Python API**: Usable as a command-line tool or integrated as a Python module.

## Installation

```bash
pip install -r requirements.txt
```

## CLI Usage

```bash
python -m src.main --input input_image.png --output output_dir/
```

Options:
- `-t, --threshold-method`: otsu, adaptive_gaussian, adaptive_mean, canny
- `-c, --coord-space`: pixel, normalized, cartesian_pixel, cartesian_normalized
- `-e, --epsilon-factor`: Simplification aggressiveness (default 0.005)
- `--no-debug`: Disable output of debug visualization PNGs.

## Python API

```python
from src import VectorizationPipeline, VectorizationConfig

config = VectorizationConfig(epsilon_factor=0.01)
pipeline = VectorizationPipeline(config)

result = pipeline.process_image("path/to/image.png", output_dir="output/")
print(f"Extracted {result.metrics.total_contours} contours.")
```
