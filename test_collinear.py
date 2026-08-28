import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig

img = cv2.imread("input/test_curve.png")

# Baseline: epsilon=0.001, collinear=1.0 deg
config1 = VectorizationConfig(epsilon_factor=0.001, collinear_angle_threshold_deg=1.0, debug_visualization=False)
res1 = VectorizationPipeline(config1).process_image(img)
print(f"Collinear 1.0: {res1.contours[0].simplified_point_count} points")

config2 = VectorizationConfig(epsilon_factor=0.001, collinear_angle_threshold_deg=0.5, debug_visualization=False)
res2 = VectorizationPipeline(config2).process_image(img)
print(f"Collinear 0.5: {res2.contours[0].simplified_point_count} points")

config3 = VectorizationConfig(epsilon_factor=0.001, remove_collinear=False, debug_visualization=False)
res3 = VectorizationPipeline(config3).process_image(img)
print(f"No collinear: {res3.contours[0].simplified_point_count} points")

