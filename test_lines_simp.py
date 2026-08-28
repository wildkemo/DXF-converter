import cv2
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig, FilterMethod
from src.geometry import simplify_contour, remove_duplicate_vertices, remove_collinear_points

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(filter_method=FilterMethod.NONE)
_, _, binary = apply_preprocessing(img, config)
raw_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
c = raw_contours[2]
print("Raw line pts:", len(c))
c2 = simplify_contour(c, 0.005, None, True)
print("Simp line pts:", len(c2))
c3 = remove_duplicate_vertices(c2)
print("No dup pts:", len(c3))
c4 = remove_collinear_points(c3, 1.0, True)
print("No collin pts:", len(c4))
