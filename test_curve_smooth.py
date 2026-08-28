import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
pts = np.array([[(x, int(100 + 50 * np.sin(x / 20.0)))] for x in range(20, 180)], dtype=np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 2)
cv2.imwrite("input/test_curve.png", img)

# Test 1: epsilon 0.001, collinear 1.0
config1 = VectorizationConfig(epsilon_factor=0.001, collinear_angle_threshold_deg=1.0, debug_visualization=False)
res1 = VectorizationPipeline(config1).process_image(img)
print(f"Epsilon 0.001, collin 1.0: {res1.contours[0].simplified_point_count} pts")

# Test 2: epsilon 0.0001, collinear 1.0
config2 = VectorizationConfig(epsilon_factor=0.0001, collinear_angle_threshold_deg=1.0, debug_visualization=False)
res2 = VectorizationPipeline(config2).process_image(img)
print(f"Epsilon 0.0001, collin 1.0: {res2.contours[0].simplified_point_count} pts")

# Test 3: epsilon 0.0001, collinear 0.1
config3 = VectorizationConfig(epsilon_factor=0.0001, collinear_angle_threshold_deg=0.1, debug_visualization=False)
res3 = VectorizationPipeline(config3).process_image(img)
print(f"Epsilon 0.0001, collin 0.1: {res3.contours[0].simplified_point_count} pts")

# Test 4: epsilon 0.0, remove_collinear=False
config4 = VectorizationConfig(epsilon_factor=0.0, remove_collinear=False, debug_visualization=False)
res4 = VectorizationPipeline(config4).process_image(img)
print(f"Epsilon 0.0, no collin: {res4.contours[0].simplified_point_count} pts")
