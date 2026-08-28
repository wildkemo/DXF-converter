import time
import os
from typing import Union, Optional
import numpy as np

from src.config import VectorizationConfig
from src.models import VectorizationResult, ImageMetadata, VectorizationMetrics
from src.preprocessing import load_and_validate_image, apply_preprocessing
from src.detection import extract_and_process_contours
from src.visualization import create_debug_views


class VectorizationPipeline:
    def __init__(self, config: Optional[VectorizationConfig] = None):
        self.config = config or VectorizationConfig()
        
    def process_image(self, image_source: Union[str, bytes, np.ndarray], output_dir: Optional[str] = None) -> VectorizationResult:
        """Run the full vectorization pipeline."""
        start_time = time.time()
        
        # 1. Load Image
        bgr_image = load_and_validate_image(image_source)
        height, width, channels = bgr_image.shape
        
        filename = "unknown"
        if isinstance(image_source, str):
            filename = os.path.basename(image_source)
            
        metadata = ImageMetadata(
            filename=filename,
            width=width,
            height=height,
            channels=channels,
            coordinate_space=self.config.coordinate_space.value
        )
        
        # 2. Preprocess
        gray, filtered, binary = apply_preprocessing(bgr_image, self.config)
        
        # 3. Detect and Process Contours
        contours = extract_and_process_contours(binary, width, height, self.config)
        
        # 4. Compute Metrics
        metrics = VectorizationMetrics()
        metrics.total_contours = len(contours)
        for c in contours:
            if c.is_hole:
                metrics.hole_contours += 1
            else:
                metrics.outer_contours += 1
                
            metrics.original_vertices += c.raw_point_count
            metrics.simplified_vertices += c.simplified_point_count
            
        metrics.processing_time_ms = (time.time() - start_time) * 1000.0
        
        result = VectorizationResult(metadata=metadata, metrics=metrics, contours=contours)
        
        # 5. Debug Visualization
        if self.config.debug_visualization and output_dir:
            create_debug_views(bgr_image, gray, filtered, binary, result, output_dir)
            
        return result
