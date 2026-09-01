import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.config import VectorizationConfig
from src.models import Contour, Point, BoundingBox
from src.geometry import simplify_contour, remove_duplicate_vertices, remove_collinear_points, normalize_winding_order, transform_coordinates


def extract_and_process_contours(binary_image: np.ndarray, width: int, height: int, config: VectorizationConfig) -> List[Contour]:
    """Extract contours, build hierarchy, filter, simplify, and construct data models."""
    
    # 1. Extraction
    raw_contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if not raw_contours or hierarchy is None:
        return []
        
    hierarchy = hierarchy[0]  # shape is (1, N, 4)
    
    # 2. Build initial tree and compute levels
    nodes = []
    for i in range(len(raw_contours)):
        h = hierarchy[i]
        nodes.append({
            'id': i,
            'next': int(h[0]),
            'prev': int(h[1]),
            'first_child': int(h[2]),
            'parent_id': int(h[3]) if h[3] != -1 else None,
            'children_ids': [],
            'level': 0,
            'raw_points': raw_contours[i],
            'valid': True
        })
        
    # Populate children lists
    for i, node in enumerate(nodes):
        parent_id = node['parent_id']
        if parent_id is not None:
            nodes[parent_id]['children_ids'].append(i)
            
    # Calculate levels (depth in tree)
    def calculate_levels(node_id, current_level):
        nodes[node_id]['level'] = current_level
        for child_id in nodes[node_id]['children_ids']:
            calculate_levels(child_id, current_level + 1)
            
    # Start level calculation from roots
    for i, node in enumerate(nodes):
        if node['parent_id'] is None:
            calculate_levels(i, 0)

    # 3. Filtering
    image_area = width * height
    for i, node in enumerate(nodes):
        raw_pts = node['raw_points']
        
        area = cv2.contourArea(raw_pts)
        perimeter = cv2.arcLength(raw_pts, config.force_closed)
        
        if area < config.min_area and perimeter < config.min_perimeter:
            node['valid'] = False
            continue
            
        if area >= config.max_area_ratio * image_area:
            node['valid'] = False
            continue
            
        if config.filter_border_touching:
            x, y, w, h_box = cv2.boundingRect(raw_pts)
            if x <= 1 or y <= 1 or (x + w) >= width - 1 or (y + h_box) >= height - 1:
                # Bounding box touches edge, might be a frame artifact
                # Let's check area ratio again to be safe, or just filter it.
                if area > 0.5 * image_area:
                     node['valid'] = False
                     continue
                     
    # 4. Reparenting orphaned children
    # Find the closest valid ancestor for each node
    for i, node in enumerate(nodes):
        # Determine topological depth
        level = 0
        curr_parent = hierarchy[i][3]
        while curr_parent != -1:
            level += 1
            curr_parent = hierarchy[curr_parent][3]
            
        # Refine Canny double-lines in Direct Color Mode
        # The inner trace of the 1px Canny line always occupies the odd topological levels.
        # We discard them to ensure a single, clean polyline per topological edge.
        if not config.use_grayscale and level % 2 != 0:
            node['valid'] = False
        
        if not node['valid']:
            continue
            
        curr_parent = node['parent_id']
        while curr_parent is not None and not nodes[curr_parent]['valid']:
            curr_parent = nodes[curr_parent]['parent_id']
            
        node['parent_id'] = curr_parent
        
    # Rebuild children_ids based on the new valid parent relationships
    for node in nodes:
        node['children_ids'] = []
        
    for i, node in enumerate(nodes):
        if node['valid'] and node['parent_id'] is not None:
            nodes[node['parent_id']]['children_ids'].append(i)

    # Re-calculate levels after re-parenting
    for i, node in enumerate(nodes):
        if node['parent_id'] is None and node['valid']:
            calculate_levels(i, 0)

    # 5. Geometry Processing and Construction
    final_contours = []
    
    for i, node in enumerate(nodes):
        if not node['valid']:
            continue
            
        raw_pts = node['raw_points']
        is_closed = config.force_closed
        
        # Simplify
        pts = simplify_contour(raw_pts, config.epsilon_factor, config.epsilon_absolute, is_closed)
        
        # Cleanup
        pts = remove_duplicate_vertices(pts)
        if config.remove_collinear:
            pts = remove_collinear_points(pts, config.collinear_angle_threshold_deg, is_closed)
            
        if len(pts) < 2:
            continue # Invalid shape
            
        # Is hole based on level (even = outer, odd = hole)
        is_hole = (node['level'] % 2) != 0
        
        # Winding order
        pts = normalize_winding_order(pts, is_hole)
        
        # Stats
        final_area = abs(cv2.contourArea(pts, oriented=False))
        final_perimeter = cv2.arcLength(pts, is_closed)
        
        # Transform coords
        model_points = transform_coordinates(pts, width, height, config.coordinate_space)
        
        # Bounding Box (computed after coordinate transformation)
        x_coords = [p.x for p in model_points]
        y_coords = [p.y for p in model_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        bbox = BoundingBox(
            x=min_x, 
            y=min_y, 
            width=max_x - min_x, 
            height=max_y - min_y
        )
        
        contour_model = Contour(
            id=i,
            parent_id=node['parent_id'],
            children_ids=[cid for cid in node['children_ids'] if nodes[cid]['valid']],
            hierarchy_level=node['level'],
            is_hole=is_hole,
            is_closed=is_closed,
            points=model_points,
            raw_point_count=len(raw_pts),
            simplified_point_count=len(pts),
            area=float(final_area),
            perimeter=float(final_perimeter),
            bounding_box=bbox
        )
        
        final_contours.append(contour_model)
        
    return final_contours
