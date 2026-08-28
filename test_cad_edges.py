import cv2
import numpy as np

# White background
img = np.ones((200, 200, 3), dtype=np.uint8) * 255

# Dark shape
cv2.rectangle(img, (20, 20), (180, 180), (30, 30, 30), -1)

# Polygons and lines INSIDE the dark shape
cv2.circle(img, (100, 100), 40, (100, 100, 100), -1) # lighter gray circle
cv2.line(img, (40, 40), (160, 160), (255, 0, 0), 2)  # Blue line

# Edges via Canny on BGR directly
edges = cv2.Canny(img, 20, 60)
cv2.imwrite("output/test_cad_edges.png", edges)

c, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(f"Contours found: {len(c)}")
