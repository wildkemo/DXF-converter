import cv2
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(debug_visualization=False)
pipeline = VectorizationPipeline(config)
res = pipeline.process_image(img)
for i, c in enumerate(res.contours):
    print(f"Contour {i}: pts={c.simplified_point_count}, area={c.area}, is_hole={c.is_hole}")
