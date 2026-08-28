import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig, FilterMethod

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(filter_method=FilterMethod.NONE, debug_visualization=False)
pipeline = VectorizationPipeline(config)
res = pipeline.process_image(img)
print(f"No filter, min_area=10: {res.metrics.total_contours} contours")

config.min_area = 0.0
pipeline2 = VectorizationPipeline(config)
res2 = pipeline2.process_image(img)
print(f"No filter, min_area=0: {res2.metrics.total_contours} contours")
