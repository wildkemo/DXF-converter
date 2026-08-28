import cv2
import numpy as np

img = np.ones((200, 200, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (20, 20), (180, 180), (30, 30, 30), -1)
cv2.circle(img, (100, 100), 40, (100, 100, 100), -1)
cv2.line(img, (40, 40), (160, 160), (255, 0, 0), 2)

edges = cv2.Canny(img, 20, 60)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

hierarchy = hierarchy[0]
valid_contours = []
for i, c in enumerate(contours):
    # If it's a hole (has a parent), wait. Canny edges have a parent if they are the inner edge.
    # Actually, the hierarchy level determines if it's the inner or outer side of the 1px line.
    # We can just check the area. The outer edge has area, the inner edge has slightly less area.
    # But Canny lines are open sometimes.
    print(f"Contour {i}: parent={hierarchy[i][3]}, pts={len(c)}")
