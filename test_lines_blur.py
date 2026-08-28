import cv2
import numpy as np
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig, FilterMethod

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(min_area=0.0)
gray, filtered, binary = apply_preprocessing(img, config)
print("Default filter unique values:", np.unique(filtered))

config2 = VectorizationConfig(filter_method=FilterMethod.NONE)
gray2, filtered2, binary2 = apply_preprocessing(img, config2)
raw_contours, _ = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f"No filter contours: {len(raw_contours)}")

