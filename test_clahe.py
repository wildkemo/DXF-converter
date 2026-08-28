import cv2
import numpy as np
from src.preprocessing import apply_preprocessing
from src.config import VectorizationConfig, ThresholdMethod

# Create an image with uneven dark illumination
img = np.zeros((100, 100, 3), dtype=np.uint8)
for x in range(100):
    for y in range(100):
        # Background gradient from 30 to 80
        bg = int(30 + 50 * (x / 100.0))
        img[y, x] = (bg, bg, bg)
        
# Draw a line that is slightly darker than the background
for i in range(10, 90):
    bg = int(30 + 50 * (i / 100.0))
    line_intensity = max(0, bg - 15) # Only 15 units darker
    img[i, i] = (line_intensity, line_intensity, line_intensity)
    img[i, i+1] = (line_intensity, line_intensity, line_intensity)

cv2.imwrite("input/test_uneven_dark.png", img)

# Test 1: Current Default (Otsu, no CLAHE)
config1 = VectorizationConfig()
gray, filtered, binary = apply_preprocessing(img, config1)
cv2.imwrite("output/test_uneven_otsu.png", binary)

# Test 2: Otsu + CLAHE
config2 = VectorizationConfig(clahe_enabled=True)
gray, filtered, binary = apply_preprocessing(img, config2)
cv2.imwrite("output/test_uneven_otsu_clahe.png", binary)

# Test 3: Adaptive Gaussian, no CLAHE
config3 = VectorizationConfig(threshold_method=ThresholdMethod.ADAPTIVE_GAUSSIAN, adaptive_block_size=21, adaptive_c=3.0)
gray, filtered, binary = apply_preprocessing(img, config3)
cv2.imwrite("output/test_uneven_adapt.png", binary)


import cv2
def count_contours(img_path):
    bin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return len(contours)

print("Otsu:", count_contours("output/test_uneven_otsu.png"))
print("Otsu+CLAHE:", count_contours("output/test_uneven_otsu_clahe.png"))
print("Adaptive:", count_contours("output/test_uneven_adapt.png"))
