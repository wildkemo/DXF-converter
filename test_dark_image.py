import cv2
import numpy as np
from src.preprocessing import detect_background_polarity, binarize_image
from src.config import VectorizationConfig, ContourPolarity

# Create a dark image (background 100, line 20)
img = np.ones((100, 100), dtype=np.uint8) * 100
cv2.line(img, (20, 20), (80, 80), (20,), 3)
cv2.imwrite("input/test_dark.png", img)

# Current polarity logic
print("Current polarity:", detect_background_polarity(img)) # Should be LIGHT_ON_DARK incorrectly

# Better logic using Otsu threshold
ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
border_median = np.median(np.concatenate([
    img[0, :], img[99, :], img[:, 0], img[:, 99]
]))
new_polarity = ContourPolarity.DARK_ON_LIGHT if border_median > ret else ContourPolarity.LIGHT_ON_DARK
print(f"Otsu Thresh: {ret}, Border median: {border_median}")
print("New polarity:", new_polarity) # Should be DARK_ON_LIGHT

