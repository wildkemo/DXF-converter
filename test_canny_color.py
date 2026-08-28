import cv2
import numpy as np

# Create an image with red triangle on green background (same luminance)
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:, :] = (0, 128, 0) # Green background
pts = np.array([[50, 20], [80, 80], [20, 80]], np.int32)
cv2.fillPoly(img, [pts], (0, 0, 128)) # Red triangle

# 1. Grayscale + Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges_gray = cv2.Canny(gray, 50, 150)
print("Gray Canny edges sum:", np.sum(edges_gray))

# 2. Color + Canny
edges_color = cv2.Canny(img, 50, 150)
print("Color Canny edges sum:", np.sum(edges_color))

cv2.imwrite("output/test_color_canny.png", edges_color)
