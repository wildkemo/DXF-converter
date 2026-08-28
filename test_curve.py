import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
# Draw a smooth sine wave
pts = np.array([[(x, int(100 + 50 * np.sin(x / 20.0)))] for x in range(20, 180)], dtype=np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 2)
cv2.imwrite("input/test_curve.png", img)

config1 = VectorizationConfig(epsilon_factor=0.005, debug_visualization=False)
res1 = VectorizationPipeline(config1).process_image(img)
print(f"Epsilon 0.005: {res1.contours[0].simplified_point_count} points")

config2 = VectorizationConfig(epsilon_factor=0.001, debug_visualization=False)
res2 = VectorizationPipeline(config2).process_image(img)
print(f"Epsilon 0.001: {res2.contours[0].simplified_point_count} points")

config3 = VectorizationConfig(epsilon_factor=0.0005, debug_visualization=False)
res3 = VectorizationPipeline(config3).process_image(img)
print(f"Epsilon 0.0005: {res3.contours[0].simplified_point_count} points")
