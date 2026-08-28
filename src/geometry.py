import math
import numpy as np
import cv2
from typing import List, Tuple, Optional
from src.models import Point
from src.config import CoordinateSpace

def simplify_contour(points_array: np.ndarray, epsilon_factor: float, epsilon_absolute: Optional[float], is_closed: bool) -> np.ndarray:
    """Simplify a contour using the Ramer-Douglas-Peucker algorithm."""
    if len(points_array) < 3:
        return points_array

    if epsilon_absolute is not None:
        epsilon = epsilon_absolute
    else:
        arc_length = cv2.arcLength(points_array, is_closed)
        epsilon = epsilon_factor * arc_length

    simplified = cv2.approxPolyDP(points_array, epsilon, is_closed)
    return simplified

def remove_duplicate_vertices(points_array: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Remove consecutive duplicate vertices."""
    if len(points_array) < 2:
        return points_array
    
    # We only check consecutive points
    pts = points_array.reshape(-1, 2)
    diff = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    mask = np.concatenate(([True], diff > tolerance))
    
    return points_array[mask]

def remove_collinear_points(points_array: np.ndarray, angle_threshold_deg: float = 1.0, is_closed: bool = True) -> np.ndarray:
    """Remove redundant vertices that lie on a straight line."""
    num_points = len(points_array)
    if num_points < 3:
        return points_array
        
    pts = points_array.reshape(-1, 2)
    keep_indices = []
    
    angle_threshold_rad = math.radians(angle_threshold_deg)
    
    for i in range(num_points):
        if i == 0:
            if not is_closed:
                keep_indices.append(i)
                continue
            p_prev = pts[-1]
        else:
            p_prev = pts[i-1]
            
        p_curr = pts[i]
        
        if i == num_points - 1:
            if not is_closed:
                keep_indices.append(i)
                continue
            p_next = pts[0]
        else:
            p_next = pts[i+1]
            
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            continue
            
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        dot_prod = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = math.acos(dot_prod)
        
        # If angle is close to 0 (vectors are parallel and same direction), it means the line is straight
        # We only keep points where angle > threshold (meaning the line bends)
        if angle > angle_threshold_rad:
            keep_indices.append(i)
            
    # Keep endpoints for open contours if they were somehow missed
    if not is_closed:
        if 0 not in keep_indices: keep_indices.insert(0, 0)
        if (num_points-1) not in keep_indices: keep_indices.append(num_points-1)
        
    if len(keep_indices) < 3 and is_closed:
        # Fallback if too many points removed and shape would become invalid
        return points_array
        
    return points_array[keep_indices]

def normalize_winding_order(points_array: np.ndarray, is_hole: bool) -> np.ndarray:
    """
    Normalize winding order.
    Outer contours -> visual CCW (OpenCV oriented area negative)
    Inner holes -> visual CW (OpenCV oriented area positive)
    """
    if len(points_array) < 3:
        return points_array
        
    area = cv2.contourArea(points_array, oriented=True)
    is_currently_cw = area > 0
    
    # Outer (not hole) should be CCW (not CW)
    # Hole should be CW
    if (not is_hole and is_currently_cw) or (is_hole and not is_currently_cw):
        # Reverse the array while preserving the shape (N, 1, 2)
        points_array = points_array[::-1]
        
    return points_array

def transform_coordinates(pts: np.ndarray, width: int, height: int, target_space: CoordinateSpace) -> List[Point]:
    """Convert points to the target coordinate space."""
    pts_reshaped = pts.reshape(-1, 2).astype(float)
    x = pts_reshaped[:, 0]
    y = pts_reshaped[:, 1]
    
    if target_space == CoordinateSpace.PIXEL:
        pass
    elif target_space == CoordinateSpace.NORMALIZED:
        x = x / width
        y = y / height
    elif target_space == CoordinateSpace.CARTESIAN_PIXEL:
        y = height - y
    elif target_space == CoordinateSpace.CARTESIAN_NORMALIZED:
        x = x / width
        y = 1.0 - (y / height)
        
    points = [Point(float(xi), float(yi)) for xi, yi in zip(x, y)]
    return points
