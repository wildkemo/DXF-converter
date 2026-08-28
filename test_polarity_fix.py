from src.preprocessing import detect_background_polarity
import numpy as np

img = np.ones((100, 100), dtype=np.uint8) * 100
print(detect_background_polarity(img))

