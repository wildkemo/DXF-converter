import math
import numpy as np
import cv2
from typing import List, Tuple, Optional
from src.models import Point
from src.config import CoordinateSpace

def smooth_contour(
    points_array: np.ndarray,
    sigma: float = 1.5,
    is_closed: bool = True,
    preserve_corners: bool = True,
    corner_threshold_deg: float = 50.0
) -> np.ndarray:
    """
    Smooth a contour using Gaussian filtering along the curve coordinates to eliminate
    discrete pixel-grid staircase/zigzag artifacts while preserving genuine sharp corners.
    """
    pts = points_array.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 5 or sigma <= 0:
        return points_array.astype(np.float32)

    k = max(2, min(5, n // 10))

    # Detect genuine sharp corners so we don't round them away
    corners = np.zeros(n, dtype=bool)
    if preserve_corners and n >= 2 * k + 1:
        if is_closed:
            p_prev = np.roll(pts, k, axis=0)
            p_next = np.roll(pts, -k, axis=0)
        else:
            p_prev = np.empty_like(pts)
            p_next = np.empty_like(pts)
            p_prev[k:] = pts[:-k]
            p_prev[:k] = pts[0]
            p_next[:-k] = pts[k:]
            p_next[-k:] = pts[-1]

        v1 = pts - p_prev
        v2 = p_next - pts
        norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
        valid = (norm1[:, 0] > 1e-3) & (norm2[:, 0] > 1e-3)
        dot = np.sum(v1 * v2, axis=1) / (norm1[:, 0] * norm2[:, 0] + 1e-9)
        dot = np.clip(dot, -1.0, 1.0)
        turning_angles = np.degrees(np.arccos(dot))

        candidate = np.zeros(n, dtype=bool)
        candidate[valid] = turning_angles[valid] > corner_threshold_deg

        # Non-maximum suppression within +/- k window
        cand_indices = np.where(candidate)[0]
        for idx in cand_indices:
            is_max = True
            val = turning_angles[idx]
            for offset in range(-k, k + 1):
                if offset == 0:
                    continue
                nbr = (idx + offset) % n if is_closed else idx + offset
                if 0 <= nbr < n:
                    if turning_angles[nbr] > val or (turning_angles[nbr] == val and offset < 0):
                        is_max = False
                        break
            if is_max:
                corners[idx] = True

        if not is_closed:
            corners[0] = True
            corners[-1] = True

    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    corner_indices = np.where(corners)[0]
    num_corners = len(corner_indices)

    # If no sharp corners, smooth the whole contour uniformly (e.g. circle, oval, smooth curve)
    if num_corners == 0:
        padded = np.pad(pts, ((half, half), (0, 0)), mode='wrap' if is_closed else 'edge')
        smoothed = np.empty_like(pts)
        smoothed[:, 0] = np.convolve(padded[:, 0], kernel, mode='valid')
        smoothed[:, 1] = np.convolve(padded[:, 1], kernel, mode='valid')
        return smoothed.reshape(-1, 1, 2).astype(np.float32)

    # If corners exist, smooth between corners, keeping corner endpoints fixed
    smoothed = pts.copy()
    for c_idx in range(num_corners):
        start_idx = corner_indices[c_idx]
        end_idx = corner_indices[(c_idx + 1) % num_corners]

        if end_idx > start_idx:
            seg = pts[start_idx:end_idx+1]
        else:
            seg = np.vstack([pts[start_idx:], pts[:end_idx+1]])

        m = len(seg)
        if m > 3:
            seg_half = min(half, (m - 1) // 2)
            if seg_half >= 1:
                seg_x = np.arange(-seg_half, seg_half + 1)
                seg_k = np.exp(-0.5 * (seg_x / sigma) ** 2)
                seg_k /= seg_k.sum()

                seg_padded = np.pad(seg, ((seg_half, seg_half), (0, 0)), mode='edge')
                seg_sm = np.empty_like(seg)
                seg_sm[:, 0] = np.convolve(seg_padded[:, 0], seg_k, mode='valid')
                seg_sm[:, 1] = np.convolve(seg_padded[:, 1], seg_k, mode='valid')

                # Lock corner endpoints
                seg_sm[0] = seg[0]
                seg_sm[-1] = seg[-1]

                if end_idx > start_idx:
                    smoothed[start_idx:end_idx+1] = seg_sm
                else:
                    split_pt = len(pts) - start_idx
                    smoothed[start_idx:] = seg_sm[:split_pt]
                    smoothed[:end_idx+1] = seg_sm[split_pt:]

    return smoothed.reshape(-1, 1, 2).astype(np.float32)

def simplify_contour(points_array: np.ndarray, epsilon_factor: float, epsilon_absolute: Optional[float], is_closed: bool) -> np.ndarray:
    """Simplify a contour using the Ramer-Douglas-Peucker algorithm."""
    if len(points_array) < 3:
        return points_array

    pts = points_array.astype(np.float32)
    if epsilon_absolute is not None:
        epsilon = epsilon_absolute
    else:
        arc_length = cv2.arcLength(pts, is_closed)
        epsilon = epsilon_factor * arc_length

    if epsilon <= 0.0:
        return pts

    simplified = cv2.approxPolyDP(pts, epsilon, is_closed)
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
        
    if len(keep_indices) < 2 and is_closed:
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

def offset_contour(points_array: np.ndarray, distance: float, is_closed: bool = True, miter_limit: float = 2.5) -> np.ndarray:
    """
    Generate an offset parallel contour shifted by `distance` for CNC toolpaths.
    Positive distance offsets outward; negative distance offsets inward.
    """
    pts = points_array.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 2 or distance == 0:
        return points_array.astype(np.float32)

    if is_closed and n < 3:
        return points_array.astype(np.float32)

    # Orientation (signed area)
    if is_closed:
        x = pts[:, 0]
        y = pts[:, 1]
        signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        # In screen coords (Y down), signed_area > 0 is CW where (-dy, dx) points inward.
        # Negate for CW so that positive distance always offsets outward consistently.
        sign = -1.0 if signed_area > 0 else 1.0
    else:
        sign = 1.0

    eff_dist = distance * sign

    if is_closed:
        p_next = np.roll(pts, -1, axis=0)
        p_prev = np.roll(pts, 1, axis=0)
    else:
        p_next = np.empty_like(pts)
        p_prev = np.empty_like(pts)
        p_next[:-1] = pts[1:]
        p_next[-1] = pts[-1] + (pts[-1] - pts[-2])
        p_prev[1:] = pts[:-1]
        p_prev[0] = pts[0] - (pts[1] - pts[0])

    v_prev = pts - p_prev
    v_next = p_next - pts

    len_prev = np.hypot(v_prev[:, 0], v_prev[:, 1])[:, None]
    len_next = np.hypot(v_next[:, 0], v_next[:, 1])[:, None]

    len_prev[len_prev < 1e-9] = 1e-9
    len_next[len_next < 1e-9] = 1e-9

    u_prev = v_prev / len_prev
    u_next = v_next / len_next

    # Normals to incoming and outgoing edges
    n_prev = np.column_stack([-u_prev[:, 1], u_prev[:, 0]])
    n_next = np.column_stack([-u_next[:, 1], u_next[:, 0]])

    # Miter bisector
    bisect = n_prev + n_next
    bisect_len = np.hypot(bisect[:, 0], bisect[:, 1])[:, None]

    valid = (bisect_len[:, 0] > 1e-4)
    bisect_unit = np.zeros_like(bisect)
    bisect_unit[valid] = bisect[valid] / bisect_len[valid]
    bisect_unit[~valid] = n_next[~valid]

    # Angle bisector scaling
    cos_half = np.sum(n_next * bisect_unit, axis=1, keepdims=True)
    cos_half = np.clip(cos_half, 0.1, 1.0)
    miter_scale = np.clip(1.0 / cos_half, 1.0, miter_limit)

    offset_pts = pts + eff_dist * miter_scale * bisect_unit
    return offset_pts.reshape(-1, 1, 2).astype(np.float32)

