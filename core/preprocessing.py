import os
from typing import Optional
import numpy as np
import cv2


#Takes any raster image (e.g., PNG, JPG).

#Converts it into a 1-bit black-and-white PBM file.

#Makes sure Potrace will see the shapes as foreground (black).

#Saves it to the specified output folder.






# take an img and convert it to .pbm file
def preprocess_to_pbm(
    input_image_path: str,
    output_dir: str,
    threshold: int = 128,
    invert: bool = True, #Useful when the background needs to be black for Potrace to detect shapes correctly.
) -> str:
    """Load image, convert to grayscale, apply binary threshold, optionally invert, and save PBM.

    Potrace considers black as foreground by default. If your shapes are white on black, keep
    invert=True so shapes become black for tracing.

    Returns the PBM file path.
    """

    # Check if the input image exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(input_image_path)

    # Create the output directory if it doesn't exist
    #exist_ok=True means that if the directory already exists, it won't raise an error.
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base name of the input image
    base = os.path.splitext(os.path.basename(input_image_path))[0] #splitext is used to split the path into a tuple of the file name and extension.
    pbm_path = os.path.join(output_dir, f"{base}.pbm")#join is used to create a full path to the PBM file.

    # Read the image in grayscale mode
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was read successfully
    if img is None:
        raise ValueError(f"Failed to read image: {input_image_path}")

    #converts it to black & white PBM format
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)#Turns the grayscale image (img) into pure black (0) and pure white (255).
    if invert:
        binary = cv2.bitwise_not(binary)#Inverts the colors of the image.

    # PBM in OpenCV uses PPM/PGM/PNM family; write binary PBM via imwrite
    success = cv2.imwrite(pbm_path, binary)#Saves the processed image in PBM format, which is the 1-bit-per-pixel monochrome format Potrace needs.
    if not success:
        raise RuntimeError(f"Failed to write PBM: {pbm_path}")

    return pbm_path


