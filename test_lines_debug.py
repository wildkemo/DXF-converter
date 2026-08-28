import cv2
import numpy as np
from src.preprocessing import binarize_image, apply_preprocessing
from src.config import VectorizationConfig

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(min_area=0.0)
gray, filtered, binary = apply_preprocessing(img, config)

cv2.imwrite("output/binary.png", binary)
raw_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f"Raw contours found: {len(raw_contours)}")
for i, c in enumerate(raw_contours):
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    print(f"Contour {i}: area={area}, perim={perim}")
