import cv2
import numpy as np
import sys
import os

img_path = "dxf coloring/output/2531-SONBOL ARNAB AWLADY.png"
if not os.path.exists(img_path):
    print(f"File {img_path} not found.")
    sys.exit(1)

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

filled_img = img.copy()
np.random.seed(42)

if hierarchy is not None:
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50:
            color = np.random.randint(50, 200, size=(3,)).tolist()
            cv2.drawContours(filled_img, [contour], 0, color, -1)

mask = thresh == 255
filled_img[mask] = [0, 0, 0]

cv2.imwrite("dxf coloring/output/filled_test.png", filled_img)
print("Saved filled_test.png")
