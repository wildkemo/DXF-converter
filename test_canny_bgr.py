import cv2
import numpy as np

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (20, 20), (180, 180), (30, 30, 30), -1) # Dark shape
cv2.circle(img, (100, 100), 40, (100, 100, 100), -1) # Inner shape
cv2.line(img, (40, 40), (160, 160), (255, 0, 0), 2)  # Blue line

edges = cv2.Canny(img, 50, 150)
cv2.imwrite("output/test_canny_bgr.png", edges)

c, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f"Canny contours: {len(c)}")
