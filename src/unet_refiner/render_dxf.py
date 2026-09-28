"""Render DXF files to binary raster images using ezdxf + OpenCV."""

import numpy as np
import cv2
import ezdxf


def render_dxf_to_image(dxf_path, size=512, line_thickness=1, padding=0.05):
    """Render a DXF file to a binary image by sampling spline curves.

    Uses ezdxf to read spline entities and approximate them as dense polylines,
    then draws them on a canvas with OpenCV for accurate, anti-aliased rendering.

    Args:
        dxf_path: Path to the DXF file.
        size: Output image size in pixels (square).
        line_thickness: Line thickness in pixels.
        padding: Fractional padding around the drawing bounding box.

    Returns:
        Float32 array (H, W) where 1.0 = ink (line), 0.0 = paper (background).
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Sample dense points from every SPLINE entity
    polylines = []
    for entity in msp:
        if entity.dxftype() != 'SPLINE':
            continue
        try:
            bspline = entity.construction_tool()
            pts = list(bspline.approximate(segments=200))
            points = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
            if len(points) < 2:
                continue
            # Close the polyline if the spline is closed
            if entity.closed and not np.allclose(points[0], points[-1]):
                points = np.vstack([points, points[:1]])
            polylines.append(points)
        except Exception:
            continue

    if not polylines:
        return np.zeros((size, size), dtype=np.float32)

    # Compute a global bounding box over all points
    all_pts = np.concatenate(polylines, axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    range_xy = max_xy - min_xy

    # Use the larger axis so the drawing isn't stretched
    max_range = max(range_xy[0], range_xy[1])
    if max_range < 1e-8:
        max_range = 1.0

    # Centre the drawing within the bounding square
    center = (min_xy + max_xy) / 2.0
    half = max_range / 2.0

    # Apply padding
    half *= (1.0 + padding)

    origin = center - half  # bottom-left of the square in DXF coords

    # White canvas
    canvas = np.ones((size, size), dtype=np.uint8) * 255

    for points in polylines:
        # Map DXF coords → pixel coords
        normalised = (points - origin) / (2.0 * half)       # [0, 1]
        px = (normalised[:, 0] * (size - 1)).astype(np.int32)
        py = ((1.0 - normalised[:, 1]) * (size - 1)).astype(np.int32)  # flip Y
        px = np.clip(px, 0, size - 1)
        py = np.clip(py, 0, size - 1)

        pixel_pts = np.stack([px, py], axis=-1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pixel_pts], isClosed=False,
                      color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    # 1.0 = ink, 0.0 = paper
    return 1.0 - (canvas.astype(np.float32) / 255.0)
