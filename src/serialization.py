import json
from dataclasses import asdict
from typing import Dict, Any, Union
import copy

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
