import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig

img = np.ones((100, 100, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (10, 10), (90, 90), (30, 30, 30), -1)
cv2.circle(img, (50, 50), 20, (100, 100, 100), -1)

config = VectorizationConfig(use_grayscale=False, debug_visualization=False)
pipeline = VectorizationPipeline(config)
result = pipeline.process_image(img)

for i, c in enumerate(result.contours):
    print(f"Contour {i}: pts={c.simplified_point_count}, area={c.area}")
