import cv2
import numpy as np

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
# Thick black donut
cv2.circle(img, (100, 100), 60, (0, 0, 0), -1)
cv2.circle(img, (100, 100), 30, (255, 255, 255), -1)

edges = cv2.Canny(img, 20, 60)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]
for i, c in enumerate(contours):
    level = 0
    parent = hierarchy[i][3]
    while parent != -1:
        level += 1
        parent = hierarchy[parent][3]
    print(f"Contour {i}: parent={hierarchy[i][3]}, level={level}, pts={len(c)}")
