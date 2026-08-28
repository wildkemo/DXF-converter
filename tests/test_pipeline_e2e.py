import cv2
import numpy as np
import pytest
from src.config import VectorizationConfig, CoordinateSpace, ThresholdMethod
from src.vectorization import VectorizationPipeline
from src.preprocessing import binarize_image, load_and_validate_image
from src.detection import extract_and_process_contours
from src.geometry import remove_collinear_points

def test_load_and_validate_image_rgba():
    # Create RGBA image with transparent background
    img = np.zeros((10, 10, 4), dtype=np.uint8)
    img[:, :, 3] = 0  # fully transparent
    
    # Load
    loaded = load_and_validate_image(img)
    assert loaded.shape == (10, 10, 3)
    # Should be blended with white
    assert np.all(loaded == 255)

def test_remove_collinear_points():
    pts = np.array([[0, 0], [10, 0], [20, 0], [20, 10], [20, 20], [0, 20]], dtype=np.int32)
    cleaned = remove_collinear_points(pts, angle_threshold_deg=1.0, is_closed=True)
    # Should remove [10, 0] and [20, 10]
    assert len(cleaned) == 4

def test_pipeline_end_to_end():
    # Create synthetic test image (white background, black square with a white hole)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), -1)
    cv2.rectangle(img, (40, 40), (60, 60), (255, 255, 255), -1)
    
    config = VectorizationConfig(
        threshold_method=ThresholdMethod.OTSU,
        coordinate_space=CoordinateSpace.PIXEL,
        epsilon_factor=0.001,
        debug_visualization=False
    )
    
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    assert result.metrics.total_contours == 2
    assert result.metrics.outer_contours == 1
    assert result.metrics.hole_contours == 1
    
    # Outer contour
    outer = next(c for c in result.contours if not c.is_hole)
    assert outer.bounding_box.width == 60
    assert outer.bounding_box.height == 60
    
    # Inner contour
    inner = next(c for c in result.contours if c.is_hole)
    assert 19 <= inner.bounding_box.width <= 25
    assert 19 <= inner.bounding_box.height <= 25
    assert inner.parent_id == outer.id
    assert inner.id in outer.children_ids

def test_pipeline_thin_lines():
    # Create an image with a single 1px thin straight line
    import cv2
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.line(img, (20, 50), (80, 50), (0, 0, 0), 1)
    
    config = VectorizationConfig(
        threshold_method=ThresholdMethod.OTSU,
        coordinate_space=CoordinateSpace.PIXEL,
        epsilon_factor=0.005,
        debug_visualization=False,
        min_area=10.0,
        min_perimeter=10.0
    )
    
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    # 1px line should be detected as a contour with 2 points and 0 area
    assert result.metrics.total_contours == 1
    contour = result.contours[0]
    assert contour.simplified_point_count == 2
    assert contour.area == 0.0
    assert contour.perimeter > 100.0  # (length ~60 * 2 = 120)
