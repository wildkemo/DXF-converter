import cv2
import numpy as np
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig, ThresholdMethod

# Create a solid black square on a white background with a gradient
img = np.zeros((200, 200, 3), dtype=np.uint8)
for x in range(200):
    for y in range(200):
        bg = int(100 + 100 * (x / 200.0))
        img[y, x] = (bg, bg, bg)
cv2.rectangle(img, (50, 50), (150, 150), (20, 20, 20), -1)
cv2.imwrite("input/test_gradient_square.png", img)

config = VectorizationConfig(threshold_method=ThresholdMethod.ADAPTIVE_GAUSSIAN, adaptive_block_size=41, adaptive_c=3.0)
_, _, binary = apply_preprocessing(img, config)
cv2.imwrite("output/test_gradient_adapt.png", binary)
