import cv2
import numpy as np

img = cv2.imread("input/test_yellow.png")

# Find background color from borders
h, w = img.shape[:2]
borders = np.concatenate([img[0, :], img[h-1, :], img[:, 0], img[:, w-1]])
borders_1d = borders[:, 0].astype(np.uint32) << 16 | borders[:, 1].astype(np.uint32) << 8 | borders[:, 2].astype(np.uint32)
unique, counts = np.unique(borders_1d, return_counts=True)
bg_1d = unique[np.argmax(counts)]
bg_color = np.array([(bg_1d >> 16) & 255, (bg_1d >> 8) & 255, bg_1d & 255], dtype=np.float32)

print("Detected background BGR:", bg_color)

# Compute Euclidean distance
dist = np.linalg.norm(img.astype(np.float32) - bg_color, axis=-1)

# Threshold distance
binary = (dist > 30.0).astype(np.uint8) * 255
cv2.imwrite("output/test_color_dist.png", binary)

c, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("Color distance contours:", len(c))
