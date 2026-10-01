import cv2
import numpy as np
import pytest
import ezdxf
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
    
    # Because of Canny double-line elimination, it will detect exactly the true shapes:
    # the square outer boundary, and the circle hole boundary. Total = 2 contours.
    assert result.metrics.total_contours == 2
    
    # Verify absolute smoothness (0.0 epsilon_factor)
    # A circle of radius 20 has perimeter ~ 125.
    # Without simplification, it should have > 50 vertices (collinear points are still removed).
    circle_contours = [c for c in result.contours if c.simplified_point_count > 50]
    assert len(circle_contours) >= 1

def test_svg_export(tmp_path):
    import cv2
    import numpy as np
    from src.serialization import to_svg
    
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), -1)
    
    config = VectorizationConfig(use_grayscale=False)
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    svg_path = str(tmp_path / "test.svg")
    svg_str = to_svg(result, svg_path)
    
    assert '<?xml version="1.0"' in svg_str
    assert '<svg width="100" height="100"' in svg_str
    assert '<polygon points="' in svg_str
    
    with open(svg_path, 'r') as f:
        content = f.read()
        assert content == svg_str

def test_dxf_export(tmp_path):
    import cv2
    import numpy as np
    import json
    from src.serialization import to_json
    from src.dxf_exporter import export_dxf
    
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), -1)
    
    config = VectorizationConfig(use_grayscale=False)
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)
    
    json_path = str(tmp_path / "test.json")
    to_json(result, output_path=json_path)
    
    dxf_path = str(tmp_path / "test.dxf")
    export_dxf(json_path, dxf_path)
    
    import os
    assert os.path.exists(dxf_path)
    assert os.path.getsize(dxf_path) > 0

def test_dxf_export_produces_splines(tmp_path):
    import cv2
    import numpy as np
    import json
    from src.serialization import to_json
    from src.dxf_exporter import export_dxf

    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), -1)

    config = VectorizationConfig(use_grayscale=False)
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    json_path = str(tmp_path / "test.json")
    to_json(result, output_path=json_path)

    dxf_path = str(tmp_path / "test.dxf")
    export_dxf(json_path, dxf_path)

    # Re-read and verify entity types
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entity_types = [e.dxftype() for e in msp]
    assert 'SPLINE' in entity_types
    assert 'LWPOLYLINE' not in entity_types

def test_save_dxf_produces_splines(tmp_path):
    import cv2
    import numpy as np
    from src.main import save_dxf

    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), -1)

    config = VectorizationConfig(use_grayscale=False)
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    dxf_path = str(tmp_path / "test_main.dxf")
    save_dxf(result, dxf_path)

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entity_types = [e.dxftype() for e in msp]
    assert 'SPLINE' in entity_types
    assert 'LWPOLYLINE' not in entity_types

def test_arc_smoothness_no_zigzag(tmp_path):
    import cv2
    import numpy as np
    import ezdxf
    from ezdxf.path import make_path
    from src.main import save_dxf

    # Create image with a smooth circle / arc of radius 80 at (150, 150)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        smooth_contours=True,
        smooth_sigma=1.5,
        epsilon_factor=0.003,
        debug_visualization=False
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    assert result.metrics.total_contours >= 1
    c = result.contours[0]

    # Save to DXF
    dxf_path = str(tmp_path / "smooth_arc.dxf")
    save_dxf(result, dxf_path)

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    spline = list(msp)[0]

    # Evaluate the spline path to check for smoothness
    p = list(make_path(spline).flattening(0.1))
    radii = [np.hypot(pt.x - 150, pt.y - 150) for pt in p]

    # Standard deviation of radius on circle must be very small (< 0.5 px)
    assert np.std(radii) < 0.5
    assert 78.5 <= min(radii) <= 80.5
    assert 79.5 <= max(radii) <= 81.5

    # Check for direction oscillations (zigzags) along the evaluated curve
    diffs = np.diff(radii)
    oscillations = np.sum(diffs[:-1] * diffs[1:] < 0)
    # A raw staircase spline has >200 zigzags; smooth arc should have <25
    assert oscillations < 25

def test_corner_preservation():
    import cv2
    import numpy as np

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        smooth_contours=True,
        smooth_sigma=1.5,
        corner_preservation=True,
        corner_threshold_deg=50.0,
        epsilon_factor=0.005,
        debug_visualization=False
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    assert result.metrics.total_contours == 1
    contour = result.contours[0]

    # Corners should be preserved, simplifying down to 4 vertices
    assert contour.simplified_point_count == 4
    assert contour.bounding_box.width == 100.0
    assert contour.bounding_box.height == 100.0


def test_cnc_line_duplication_outward():
    import cv2
    import numpy as np

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        epsilon_factor=0.005,
        debug_visualization=False,
        duplicate_distance=5.0,
        duplicate_both_sides=False
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    assert result.metrics.total_contours == 2
    original = result.contours[0]
    duplicate = result.contours[1]

    assert not original.is_duplicate
    assert duplicate.is_duplicate
    assert duplicate.offset_distance == 5.0
    assert duplicate.original_contour_id == original.id

    # The duplicated contour must be shifted outward by 5.0 in all directions (width + 10, height + 10)
    assert abs(duplicate.bounding_box.width - (original.bounding_box.width + 10.0)) < 1.0
    assert abs(duplicate.bounding_box.height - (original.bounding_box.height + 10.0)) < 1.0
    assert duplicate.area > original.area


def test_cnc_line_duplication_both_sides():
    import cv2
    import numpy as np

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        epsilon_factor=0.005,
        debug_visualization=False,
        duplicate_distance=4.0,
        duplicate_both_sides=True
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    # 1 original + 2 duplicates (+4.0 outward, -4.0 inward) = 3 contours
    assert result.metrics.total_contours == 3
    orig = result.contours[0]
    dup_out = next(c for c in result.contours if c.is_duplicate and c.offset_distance == 4.0)
    dup_in = next(c for c in result.contours if c.is_duplicate and c.offset_distance == -4.0)

    assert dup_out.area > orig.area > dup_in.area
    assert dup_out.original_contour_id == orig.id
    assert dup_in.original_contour_id == orig.id


def test_cnc_dxf_layers_and_colors(tmp_path):
    import cv2
    import numpy as np
    from src.main import save_dxf
    from src.serialization import to_json
    from src.dxf_exporter import export_dxf

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        debug_visualization=False,
        duplicate_distance=3.0
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    # Test main.save_dxf
    dxf_main = str(tmp_path / "main_cnc.dxf")
    save_dxf(result, dxf_main)
    doc_main = ezdxf.readfile(dxf_main)
    assert "CNC_OFFSET" in doc_main.layers
    assert doc_main.layers.get("CNC_OFFSET").color == 4  # Cyan

    splines = list(doc_main.modelspace().query("SPLINE"))
    assert len(splines) == 2
    offset_splines = [s for s in splines if s.dxf.layer == "CNC_OFFSET"]
    assert len(offset_splines) == 1

    # Test dxf_exporter.export_dxf
    json_path = str(tmp_path / "contours.json")
    to_json(result, output_path=json_path)
    dxf_exporter = str(tmp_path / "exported_cnc.dxf")
    export_dxf(json_path, dxf_exporter)
    doc_exp = ezdxf.readfile(dxf_exporter)
    assert "CNC_OFFSET" in doc_exp.layers
    assert doc_exp.layers.get("CNC_OFFSET").color == 4


def test_cnc_svg_export(tmp_path):
    import cv2
    import numpy as np
    from src.serialization import to_svg

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.OTSU,
        debug_visualization=False,
        duplicate_distance=3.0
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    svg_path = str(tmp_path / "cnc.svg")
    svg_str = to_svg(result, svg_path)

    assert 'id="cnc_offset_contours"' in svg_str
    assert 'stroke="#0080ff"' in svg_str
    assert 'id="original_contours"' in svg_str


def test_enhanced_detection_hybrid_all():
    import cv2
    import numpy as np

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Dark rectangle
    cv2.rectangle(img, (20, 20), (80, 80), (30, 30, 30), -1)
    # Subtle colored line
    cv2.line(img, (100, 20), (180, 20), (0, 180, 220), 2)
    # Circle
    cv2.circle(img, (140, 120), 30, (80, 80, 80), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        threshold_method=ThresholdMethod.HYBRID_ALL,
        debug_visualization=False,
        min_area=5.0,
        min_perimeter=5.0
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    # All 3 distinct shapes should be captured cleanly
    assert result.metrics.total_contours >= 3


def test_serialization_roundtrip_with_duplicates(tmp_path):
    import cv2
    import numpy as np
    from src.serialization import to_json, from_json

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    config = VectorizationConfig(
        use_grayscale=True,
        debug_visualization=False,
        duplicate_distance=2.5
    )
    pipeline = VectorizationPipeline(config)
    result = pipeline.process_image(img)

    json_path = str(tmp_path / "roundtrip.json")
    to_json(result, output_path=json_path)

    loaded_result = from_json(json_path)
    assert len(loaded_result.contours) == 2
    dup = loaded_result.contours[1]
    assert dup.is_duplicate is True
    assert dup.offset_distance == 2.5
    assert dup.original_contour_id == loaded_result.contours[0].id

