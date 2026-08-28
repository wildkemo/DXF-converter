# Image Contour Detection and Vectorization Implementation Plan

## 1. Objective

Build a Python-based system that takes a raster image containing primarily 2D illustrations and converts the visual content into vector contours.

The system must not attempt to understand the semantic meaning of the objects. Its purpose is geometric extraction: identify visible boundaries and shapes in the image, represent those boundaries as vector paths, and return the resulting contours as ordered coordinate points.

The final output of the system is a structured collection of vector contours represented by their coordinates. JSON should be used as the initial output format because it is simple to inspect, debug, and consume from other applications.

The system should be capable of detecting both outer and internal contours. For example, if an image contains a large rectangle with circles, holes, lines, and smaller shapes inside it, the result should contain separate vector representations for those structures rather than only the outermost boundary.

The initial implementation should focus on reliable contour extraction and vectorization. More advanced geometric fitting, such as identifying perfect circles or arcs, should be treated as later improvements.

---

# 2. Scope

The system should support raster images such as:

- Simple black-and-white illustrations
- Line drawings
- Geometric illustrations
- Diagrams
- Icons
- Scanned 2D drawings
- Illustrations containing multiple independent shapes
- Illustrations containing nested shapes
- Illustrations containing holes or enclosed regions
- Illustrations containing both filled shapes and outlined shapes

The system should attempt to preserve:

- The number of meaningful contours
- The shape of each contour
- The relative position of contours
- The contour hierarchy
- The ordering of points along each contour

The system does not need to:

- Recognize what an object represents
- Classify objects
- Generate semantic labels
- Produce a 3D model
- Generate DXF
- Generate SVG
- Detect objects using a pretrained object detector

The output at this stage is only vector contour geometry.

---

# 3. Technology Stack

## Programming Language

Use:

- Python 3.11 or newer

Python is recommended because OpenCV and the surrounding image-processing ecosystem are mature and well suited to this task.

## Required Dependencies

Install exactly these packages for the initial implementation:

- `opencv-python >= 4.10`
- `numpy >= 1.26`

OpenCV provides image loading, preprocessing, thresholding, edge detection, contour extraction, contour approximation, morphology, and geometric measurements.

NumPy is used for efficient representation and manipulation of image data and coordinate arrays.

## Optional Dependencies

The following packages should not be required for the first prototype, but can be introduced when more advanced processing is needed:

- `shapely >= 2.0`
- `scikit-image >= 0.24`

Shapely can be used later for geometric validation, polygon operations, simplification, intersection detection, and cleanup.

scikit-image can provide alternative image-processing algorithms when OpenCV alone is insufficient.

## Installation

The project's dependency file should contain the exact package requirements used by the implementation.

The initial environment should therefore contain:

- Python 3.11+
- OpenCV
- NumPy

Do not add machine-learning frameworks unless testing later demonstrates that classical computer vision is insufficient.

---

# 4. High-Level Architecture

The complete pipeline should be organized into the following stages:

1. Input image loading
2. Image validation
3. Image preprocessing
4. Segmentation or edge extraction
5. Morphological cleanup when necessary
6. Contour detection
7. Contour hierarchy extraction
8. Contour filtering
9. Contour simplification
10. Optional geometric cleanup
11. Coordinate normalization if required
12. Vector contour construction
13. JSON serialization
14. Debug visualization
15. Quality evaluation

The conceptual pipeline is:

Input Image

→ Preprocessing

→ Segmentation / Edge Detection

→ Contour Detection

→ Filtering

→ Simplification

→ Geometric Cleanup

→ Vector Coordinates

→ JSON Output

---

# 5. Project Structure

Organize the project into separate responsibilities rather than putting the entire pipeline into one script.

Recommended structure:

```text
contour-vectorizer/
├── input/
│   └── image.png
├── output/
│   ├── contours.json
│   ├── debug_preprocessed.png
│   └── debug_contours.png
├── src/
│   ├── preprocessing.py
│   ├── detection.py
│   ├── vectorization.py
│   ├── geometry.py
│   ├── serialization.py
│   └── main.py
├── tests/
├── requirements.txt
└── README.md
```
