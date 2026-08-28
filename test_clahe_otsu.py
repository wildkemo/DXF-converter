import cv2
import numpy as np
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig

# Very low contrast image: background 80, lines 70
img = np.ones((100, 100, 3), dtype=np.uint8) * 80
cv2.rectangle(img, (20, 20), (80, 80), (70, 70, 70), -1)

config1 = VectorizationConfig(clahe_enabled=False)
_, _, bin1 = apply_preprocessing(img, config1)

config2 = VectorizationConfig(clahe_enabled=True)
_, _, bin2 = apply_preprocessing(img, config2)

def check_contours(binary):
    c, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return len(c)

print(f"No CLAHE contours: {check_contours(bin1)}")
print(f"CLAHE contours: {check_contours(bin2)}")
