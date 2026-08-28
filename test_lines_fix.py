import cv2
import numpy as np
from src.vectorization import VectorizationPipeline
from src.config import VectorizationConfig, FilterMethod

img = cv2.imread("input/test_lines.png")
config = VectorizationConfig(filter_method=FilterMethod.NONE, debug_visualization=False)

# Hack the classes for testing
import src.detection
old_extract = src.detection.extract_and_process_contours

def new_extract(*args, **kwargs):
    # Just run the pipeline but we will patch detection.py manually for this test
    pass

