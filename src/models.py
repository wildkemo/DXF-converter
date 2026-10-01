from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Point:
    x: float
    y: float
    
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0
        
    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class Contour:
    id: int
    parent_id: Optional[int]
    children_ids: List[int]
    hierarchy_level: int
    is_hole: bool
    is_closed: bool
    points: List[Point]
    raw_point_count: int
    simplified_point_count: int
    area: float
    perimeter: float
    bounding_box: BoundingBox
    is_duplicate: bool = False
    offset_distance: Optional[float] = None
    original_contour_id: Optional[int] = None

@dataclass
class ImageMetadata:
    filename: str
    width: int
    height: int
    channels: int
    coordinate_space: str

@dataclass
class VectorizationMetrics:
    total_contours: int = 0
    outer_contours: int = 0
    hole_contours: int = 0
    original_vertices: int = 0
    simplified_vertices: int = 0
    processing_time_ms: float = 0.0
    
    @property
    def vertex_reduction_pct(self) -> float:
        if self.original_vertices == 0:
            return 0.0
        return (1.0 - (self.simplified_vertices / self.original_vertices)) * 100.0

@dataclass
class VectorizationResult:
    metadata: ImageMetadata
    metrics: VectorizationMetrics
    contours: List[Contour]
