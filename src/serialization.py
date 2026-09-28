import json
from dataclasses import asdict
from typing import Dict, Any, Union
import copy
import cv2
import numpy as np

from src.models import VectorizationResult, Contour, Point, BoundingBox, ImageMetadata, VectorizationMetrics


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)


def to_json(result: VectorizationResult, pretty: bool = True, output_path: str = None) -> str:
    """Serialize VectorizationResult to JSON."""
    indent = 4 if pretty else None
    
    # Use asdict manually to handle enums and floats properly if needed
    data = asdict(result)
    
    json_str = json.dumps(data, indent=indent, cls=CustomJSONEncoder)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
    return json_str


def from_json(json_source: Union[str, Dict[str, Any]]) -> VectorizationResult:
    """Deserialize JSON back into VectorizationResult."""
    if isinstance(json_source, str):
        # Could be a path or a JSON string
        try:
            with open(json_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, IOError, FileNotFoundError):
            data = json.loads(json_source)
    else:
        data = copy.deepcopy(json_source)
        
    metadata = ImageMetadata(**data['metadata'])
    metrics = VectorizationMetrics(**data['metrics'])
    
    contours = []
    for c_data in data['contours']:
        points = [Point(**p) for p in c_data['points']]
        bbox = BoundingBox(**c_data['bounding_box'])
        
        c = Contour(
            id=c_data['id'],
            parent_id=c_data['parent_id'],
            children_ids=c_data['children_ids'],
            hierarchy_level=c_data['hierarchy_level'],
            is_hole=c_data['is_hole'],
            is_closed=c_data['is_closed'],
            points=points,
            raw_point_count=c_data['raw_point_count'],
            simplified_point_count=c_data['simplified_point_count'],
            area=c_data['area'],
            perimeter=c_data['perimeter'],
            bounding_box=bbox
        )
        contours.append(c)
        
    return VectorizationResult(metadata=metadata, metrics=metrics, contours=contours)

def to_svg(result: VectorizationResult, output_path: str = None) -> str:
    """Serialize VectorizationResult to an SVG file."""
    w, h = result.metadata.width, result.metadata.height
    coord_space = result.metadata.coordinate_space
    
    if "normalized" in coord_space:
        viewbox = "0 0 1 1"
        stroke_width = 0.001
    else:
        viewbox = f"0 0 {w} {h}"
        stroke_width = 1.0
        
    if "cartesian" in coord_space:
        if "normalized" in coord_space:
            transform = 'transform="translate(0, 1) scale(1, -1)"'
        else:
            transform = f'transform="translate(0, {h}) scale(1, -1)"'
    else:
        transform = ""

    svg = [
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg width="{w}" height="{h}" viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">',
        f'  <rect width="100%" height="100%" fill="white" />',
        f'  <g fill="none" stroke="black" stroke-width="{stroke_width}" {transform}>'
    ]
    
    for c in result.contours:
        pts_str = " ".join([f"{p.x},{p.y}" for p in c.points])
        tag = "polygon" if c.is_closed else "polyline"
        svg.append(f'    <{tag} points="{pts_str}" />')
        
    svg.append('  </g>')
    svg.append('</svg>')
    
    svg_str = "\n".join(svg)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_str)
            
    return svg_str

def to_points_image(result: VectorizationResult, output_path: str) -> None:
    """Draw all contour points on a blank white canvas and save as an image."""
    w, h = result.metadata.width, result.metadata.height
    coord_space = result.metadata.coordinate_space
    
    # Create white canvas
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    for c in result.contours:
        for p in c.points:
            x, y = p.x, p.y
            
            # Map back to pixel space for drawing
            if "normalized" in coord_space:
                x = int(x * w)
                y = int(y * h)
            else:
                x = int(x)
                y = int(y)
                
            if "cartesian" in coord_space:
                y = h - y
                
            # Draw a 1-pixel black dot
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1)
            
    cv2.imwrite(output_path, img)
