import cv2
import numpy as np
img = np.ones((200, 200, 3), dtype=np.uint8) * 255
cv2.circle(img, (100, 100), 80, (0, 0, 0), -1) # outer circle
cv2.circle(img, (100, 100), 40, (255, 255, 255), -1) # inner hole
cv2.imwrite("input/sample.png", img)
