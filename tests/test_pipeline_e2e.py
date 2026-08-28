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
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        coordinate_space=CoordinateSpace.PIXEL,
        epsilon_factor=0.005,
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
        use_grayscale=True,
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

def test_dark_image_polarity():
    from src.preprocessing import detect_background_polarity
    from src.config import ContourPolarity
    import numpy as np
    
    # Create a dark image: dark-gray background (100) and black lines (20)
    # The overall image is much darker than the hardcoded 127 threshold.
    img = np.ones((100, 100), dtype=np.uint8) * 100
    import cv2
    cv2.line(img, (20, 20), (80, 80), (20,), 3)
    
    polarity = detect_background_polarity(img)
    # The border is 100, which is higher than the Otsu threshold (~60)
    # so it should correctly detect DARK_ON_LIGHT.
    assert polarity == ContourPolarity.DARK_ON_LIGHT

def test_pipeline_low_contrast_faint_lines():
    import cv2
    import numpy as np
    
    # Very dark background with very faint lines
    # Background: 30, Lines: 45
    # The contrast difference is only 15, very hard to see without CLAHE
    img = np.ones((100, 100, 3), dtype=np.uint8) * 30
    cv2.rectangle(img, (20, 20), (80, 80), (45, 45, 45), -1)
    
    # Run with default config (CLAHE should be True)
    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        coordinate_space=CoordinateSpace.PIXEL,
        debug_visualization=False,
        min_area=10.0,
        min_perimeter=10.0
    )
    
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    # The faint rectangle should be found!
    assert result.metrics.total_contours >= 1
    assert any(c.area > 3000 for c in result.contours)

def test_direct_color_distance():
    import cv2
    import numpy as np
    
    # Image with white background, yellow line, light-blue triangle
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.line(img, (20, 20), (80, 20), (0, 255, 255), 2) # Yellow
    
    pts = np.array([[50, 40], [80, 80], [20, 80]], np.int32)
    cv2.fillPoly(img, [pts], (255, 200, 0)) # Light blue
    
    config = VectorizationConfig(
        use_grayscale=False, # Direct color mode
        color_distance_threshold=30.0,
        debug_visualization=False
    )
    
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    # We should detect at least 2 contours: the yellow line and the blue triangle
    assert result.metrics.total_contours >= 2

def test_nested_shapes_and_smoothness():
    import cv2
    import numpy as np
    
    # White background
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Dark shape
    cv2.rectangle(img, (10, 10), (90, 90), (30, 30, 30), -1)
    # Circle INSIDE the dark shape
    cv2.circle(img, (50, 50), 20, (100, 100, 100), -1)
    
    config = VectorizationConfig(
        use_grayscale=False,
        debug_visualization=False
    )
    
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    # Because of Canny edges, it will detect the outer edge of the square,
    # the inner edge of the square, the outer edge of the circle, 
    # and the inner edge of the circle. Total = 4 contours minimum.
    assert result.metrics.total_contours >= 4
    
    # Verify absolute smoothness (0.0 epsilon_factor)
    # A circle of radius 20 has perimeter ~ 125.
    # Without simplification, it should have > 50 vertices (collinear points are still removed).
    circle_contours = [c for c in result.contours if c.simplified_point_count > 50]
    assert len(circle_contours) >= 1
