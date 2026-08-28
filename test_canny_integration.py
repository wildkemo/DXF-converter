import cv2
import numpy as np
from src.preprocessing import binarize_color_distance

img = np.ones((100, 100, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (10, 10), (90, 90), (50, 50, 50), -1)
cv2.circle(img, (50, 50), 20, (200, 50, 50), -1) # Red circle inside dark gray square

# If we used old color distance:
dist = binarize_color_distance(img)
c1, _ = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f"Color distance contours: {len(c1)} (red circle is merged/neglected!)")

# If we use direct BGR Canny:
edges = cv2.Canny(img, 50, 150)
c2, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f"Canny BGR contours: {len(c2)} (red circle is detected!)")
