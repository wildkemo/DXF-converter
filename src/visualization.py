import cv2
import numpy as np
import os
from typing import List
from src.models import VectorizationResult, Contour
from src.config import CoordinateSpace


def create_debug_views(
    original_bgr: np.ndarray, 
    preprocessed_gray: np.ndarray, 
    filtered: np.ndarray, 
    binary_mask: np.ndarray, 
    result: VectorizationResult, 
    output_dir: str
):
    """Generate debug visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = original_bgr.shape[:2]
    
    # 1. Preprocessing multi-panel
    gray_bgr = cv2.cvtColor(preprocessed_gray, cv2.COLOR_GRAY2BGR)
    filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    top_row = np.hstack((original_bgr, gray_bgr))
    bottom_row = np.hstack((filtered_bgr, binary_bgr))
    panel = np.vstack((top_row, bottom_row))
    
    # Resize if too large
    max_dim = 1600
    if panel.shape[1] > max_dim or panel.shape[0] > max_dim:
        scale = max_dim / max(panel.shape[0], panel.shape[1])
        panel = cv2.resize(panel, (0, 0), fx=scale, fy=scale)
        
    cv2.imwrite(os.path.join(output_dir, "debug_preprocessed.png"), panel)
    
    # 2. Contours debug
    # Create white canvas
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    overlay = original_bgr.copy()
    
    # If coordinate space is not PIXEL, we need to map back for visualization
    def map_to_pixel(x, y):
        cs = result.metadata.coordinate_space
        if cs == CoordinateSpace.NORMALIZED.value:
            return int(x * w), int(y * h)
        elif cs == CoordinateSpace.CARTESIAN_PIXEL.value:
            return int(x), int(h - y)
        elif cs == CoordinateSpace.CARTESIAN_NORMALIZED.value:
            return int(x * w), int(h - (y * h))
        return int(x), int(y)
        
    for contour in result.contours:
        pts = np.array([[map_to_pixel(p.x, p.y)] for p in contour.points], dtype=np.int32)
        
        # Color coding: Green for outer, Red for holes, Blue for deep nested
        if contour.is_hole:
            color = (0, 0, 255) # BGR Red
        elif contour.hierarchy_level == 0:
            color = (0, 200, 0) # Green
        else:
            color = (255, 0, 0) # Blue
            
        cv2.drawContours(canvas, [pts], -1, color, 2)
        cv2.drawContours(overlay, [pts], -1, color, 2)
        
        # Draw vertices
        for pt in pts:
            cv2.circle(canvas, tuple(pt[0]), 3, (0, 0, 0), -1)
            
        # Draw ID near first point
        if len(pts) > 0:
            pos = tuple(pts[0][0])
            cv2.putText(canvas, str(contour.id), (pos[0]+5, pos[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
    cv2.imwrite(os.path.join(output_dir, "debug_contours.png"), canvas)
    
    # Blended overlay
    blended = cv2.addWeighted(original_bgr, 0.3, overlay, 0.7, 0)
    cv2.imwrite(os.path.join(output_dir, "debug_overlay.png"), blended)
