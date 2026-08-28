from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional


class ThresholdMethod(str, Enum):
    OTSU = "otsu"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    ADAPTIVE_MEAN = "adaptive_mean"
    CANNY = "canny"
    BINARY_FIXED = "binary_fixed"


class FilterMethod(str, Enum):
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    MEDIAN = "median"
    NONE = "none"


class CoordinateSpace(str, Enum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"
    CARTESIAN_PIXEL = "cartesian_pixel"
    CARTESIAN_NORMALIZED = "cartesian_normalized"


class ContourPolarity(str, Enum):
    AUTO = "auto"
    DARK_ON_LIGHT = "dark_on_light"
    LIGHT_ON_DARK = "light_on_dark"


@dataclass
class VectorizationConfig:
    # Thresholding / Binarization
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU
    threshold_value: int = 127
    adaptive_block_size: int = 11
    adaptive_c: float = 2.0
    canny_low: Optional[float] = None
    canny_high: Optional[float] = None
    polarity: ContourPolarity = ContourPolarity.AUTO

    # Preprocessing Filters
    filter_method: FilterMethod = FilterMethod.NONE
    filter_kernel_size: int = 3
    filter_sigma: float = 1.0
    
    # Contrast Enhancement
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # Morphology
    morph_open_kernel: int = 0  # 0 disables
    morph_close_kernel: int = 0  # 0 disables

    # Filtering
    min_area: float = 10.0
    max_area_ratio: float = 0.999
    min_perimeter: float = 10.0
    filter_border_touching: bool = True

    # Geometry / Simplification
    epsilon_factor: float = 0.0001
    epsilon_absolute: Optional[float] = None
    force_closed: bool = True
    remove_collinear: bool = True
    collinear_angle_threshold_deg: float = 0.1
    
    # Output Space
    coordinate_space: CoordinateSpace = CoordinateSpace.PIXEL
    
    # Debug
    debug_visualization: bool = True
