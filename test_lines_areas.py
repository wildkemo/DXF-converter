import cv2
import numpy as np
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig, FilterMethod

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(filter_method=FilterMethod.NONE)
gray, filtered, binary = apply_preprocessing(img, config)
raw_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(raw_contours):
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    print(f"Contour {i}: area={area}, perim={perim}")
