import cv2
import numpy as np
from typing import Tuple, Union, Optional
from src.config import VectorizationConfig, ThresholdMethod, FilterMethod, ContourPolarity


def load_and_validate_image(source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """Load image from path, bytes, or array and return BGR image. Handle RGBA transparency."""
    if isinstance(source, str):
        img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image from path: {source}")
    elif isinstance(source, bytes):
        img_array = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
    elif isinstance(source, np.ndarray):
        img = source
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    if len(img.shape) == 2:
        # Grayscale to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # RGBA to BGR (blend with white background)
        alpha_channel = img[:, :, 3] / 255.0
        bgr_channels = img[:, :, :3]
        white_background = np.ones_like(bgr_channels) * 255
        
        # Blend
        blended = np.zeros_like(bgr_channels, dtype=np.float32)
        for c in range(3):
            blended[:, :, c] = (bgr_channels[:, :, c] * alpha_channel + 
                                white_background[:, :, c] * (1.0 - alpha_channel))
        img = blended.astype(np.uint8)
    
    return img


def apply_preprocessing(bgr_image: np.ndarray, config: VectorizationConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply full preprocessing pipeline.
    Returns: (gray_image, filtered_image, binary_mask)
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # 1. Contrast Enhancement
    if config.clahe_enabled:
        clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid_size)
        gray = clahe.apply(gray)
        
    # 2. Filtering
    if config.filter_method == FilterMethod.GAUSSIAN:
        filtered = cv2.GaussianBlur(gray, (config.filter_kernel_size, config.filter_kernel_size), config.filter_sigma)
    elif config.filter_method == FilterMethod.BILATERAL:
        filtered = cv2.bilateralFilter(gray, config.filter_kernel_size, config.filter_sigma * 50, config.filter_sigma * 50)
    elif config.filter_method == FilterMethod.MEDIAN:
        filtered = cv2.medianBlur(gray, config.filter_kernel_size)
    else:
        filtered = gray.copy()
        
    # 3. Binarization
    binary = binarize_image(filtered, config)
    
    # 4. Morphology
    if config.morph_open_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.morph_open_kernel, config.morph_open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
    if config.morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.morph_close_kernel, config.morph_close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
    return gray, filtered, binary


def detect_background_polarity(gray_image: np.ndarray) -> ContourPolarity:
    """Detect if the image is dark-on-light or light-on-dark based on border pixels."""
    h, w = gray_image.shape
    border_pixels = np.concatenate([
        gray_image[0, :], gray_image[h-1, :], 
        gray_image[:, 0], gray_image[:, w-1]
    ])
    median_border = np.median(border_pixels)
    
    ret, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if median_border > ret:
        return ContourPolarity.DARK_ON_LIGHT
    else:
        return ContourPolarity.LIGHT_ON_DARK


def binarize_image(gray: np.ndarray, config: VectorizationConfig) -> np.ndarray:
    """Binarize the grayscale image based on config."""
    polarity = config.polarity
    if polarity == ContourPolarity.AUTO:
        polarity = detect_background_polarity(gray)
        
    if config.threshold_method == ThresholdMethod.CANNY:
        # Canny edge detection
        if config.canny_low is None or config.canny_high is None:
            median_val = np.median(gray)
            sigma = 0.33
            low = int(max(0, (1.0 - sigma) * median_val))
            high = int(min(255, (1.0 + sigma) * median_val))
        else:
            low, high = int(config.canny_low), int(config.canny_high)
            
        edges = cv2.Canny(gray, low, high)
        return edges

    # For thresholding, we want the objects to be WHITE (255) and background BLACK (0)
    # for findContours to work properly.
    if polarity == ContourPolarity.DARK_ON_LIGHT:
        # Objects are dark, background is light. We need to invert.
        thresh_type = cv2.THRESH_BINARY_INV
    else:
        thresh_type = cv2.THRESH_BINARY

    if config.threshold_method == ThresholdMethod.OTSU:
        _, binary = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)
    elif config.threshold_method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        binary = cv2.adaptiveThreshold(gray, 255, adaptive_method, thresh_type, 
                                       config.adaptive_block_size, config.adaptive_c)
    elif config.threshold_method == ThresholdMethod.ADAPTIVE_MEAN:
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        binary = cv2.adaptiveThreshold(gray, 255, adaptive_method, thresh_type, 
                                       config.adaptive_block_size, config.adaptive_c)
    elif config.threshold_method == ThresholdMethod.BINARY_FIXED:
        _, binary = cv2.threshold(gray, config.threshold_value, 255, thresh_type)
    else:
        raise ValueError(f"Unknown threshold method: {config.threshold_method}")

    return binary
