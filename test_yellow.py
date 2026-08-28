import cv2
import numpy as np
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig

# White background, Black square, Yellow line
img = np.ones((100, 100, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (10, 10), (40, 40), (0, 0, 0), -1) # Black square
cv2.line(img, (60, 20), (60, 80), (0, 255, 255), 2) # Yellow line

cv2.imwrite("input/test_yellow.png", img)

config = VectorizationConfig(clahe_enabled=False)
gray, filtered, binary = apply_preprocessing(img, config)
cv2.imwrite("output/test_yellow_binary.png", binary)

edges = cv2.Canny(img, 50, 150)
cv2.imwrite("output/test_yellow_canny.png", edges)
