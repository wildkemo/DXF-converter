import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
# Draw a closed square
cv2.rectangle(img, (20, 20), (50, 50), (0, 0, 0), -1)
# Draw a thin line
cv2.line(img, (100, 20), (100, 150), (0, 0, 0), 1)
# Draw another thin curve
cv2.circle(img, (150, 100), 30, (0, 0, 0), 1)

cv2.imwrite("input/test_lines.png", img)

config = VectorizationConfig(min_area=10.0, debug_visualization=False)
pipeline = VectorizationPipeline(config)
res = pipeline.process_image(img)
print(f"With min_area=10: {res.metrics.total_contours} contours")

config2 = VectorizationConfig(min_area=0.0, debug_visualization=False)
pipeline2 = VectorizationPipeline(config2)
res2 = pipeline2.process_image(img)
print(f"With min_area=0: {res2.metrics.total_contours} contours")
