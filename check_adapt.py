import cv2
import numpy as np
img = cv2.imread("output/test_gradient_adapt.png", cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(contours):
    print(f"Adaptive Contour {i}: area={cv2.contourArea(c)}, perim={cv2.arcLength(c, True)}")
