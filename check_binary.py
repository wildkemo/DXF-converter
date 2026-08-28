import cv2
import numpy as np
for name in ["otsu", "otsu_clahe", "adapt"]:
    img = cv2.imread(f"output/test_uneven_{name}.png", cv2.IMREAD_GRAYSCALE)
    print(f"{name}: unique values={np.unique(img)}, mean={np.mean(img):.2f}")
