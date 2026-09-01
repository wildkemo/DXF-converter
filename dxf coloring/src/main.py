import os
import glob
import argparse
import shutil
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy

import cv2
import numpy as np

def convert_dxf_to_image(input_path, output_path):
    print(f"Processing: {input_path}")
    try:
        # Load the DXF document
        doc = ezdxf.readfile(input_path)
        msp = doc.modelspace()
        
        # Create a matplotlib figure
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        
        # Create rendering context
        ctx = RenderContext(doc)
        ctx.set_current_layout(msp)
        
        # Matplotlib backend
        out = MatplotlibBackend(ax)
        
        # Configuration for light background
        config = Configuration(
            background_policy=BackgroundPolicy.WHITE,
            color_policy=ColorPolicy.COLOR_SWAP_BW
        )
        
        # Render the DXF to the matplotlib axes
        Frontend(ctx, out, config=config).draw_layout(msp, finalize=True)
        
        # Save as a PNG file
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Fill closed shapes with colors using OpenCV
        img = cv2.imread(output_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold to find lines (invert so lines are white)
            ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of the closed shapes
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            filled_img = img.copy()
            # Set a random seed based on the file name for reproducible colors
            np.random.seed(hash(os.path.basename(input_path)) % 100000)
            
            if hierarchy is not None:
                img_area = img.shape[0] * img.shape[1]
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    # Ignore tiny artifacts and the entire outer background bounds
                    if 50 < area < img_area * 0.9:
                        color = np.random.randint(50, 220, size=(3,)).tolist()
                        cv2.drawContours(filled_img, [contour], 0, color, -1)
            
            # Redraw original lines on top to preserve the boundaries
            mask = thresh == 255
            filled_img[mask] = [0, 0, 0]
            
            # Save the final filled image
            cv2.imwrite(output_path, filled_img)
            
        print(f"Saved filled image to: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert DXF files to colored images")
    
    # Path relative to the src/ directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(base_dir, "input")
    default_output = os.path.join(base_dir, "dataset")
    
    parser.add_argument("--input-dir", type=str, default=default_input, help="Directory containing input DXF files")
    parser.add_argument("--output-dir", type=str, default=default_output, help="Directory to save output image files")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of files to process (0 for unlimited)")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .dxf files in the input directory (case-insensitive search could be added, but standard glob is fine)
    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))
    dxf_files.extend(glob.glob(os.path.join(input_dir, "*.DXF")))
    
    if not dxf_files:
        print(f"No DXF files found in {input_dir}")
        return
        
    # Sort files to ensure consistent ordering for pair 1, 2, 3...
    dxf_files = sorted(list(set(dxf_files)))
    if args.limit > 0:
        dxf_files = dxf_files[:args.limit]
        
    for i, dxf_file in enumerate(dxf_files, start=1):
        pair_dir = os.path.join(output_dir, f"pair_{i}")
        os.makedirs(pair_dir, exist_ok=True)
        
        output_image = os.path.join(pair_dir, "image.png")
        output_dxf = os.path.join(pair_dir, "perfect.dxf")
        
        # Copy the original DXF file into the pair folder
        shutil.copy(dxf_file, output_dxf)
        
        # Convert and save the colored image
        convert_dxf_to_image(dxf_file, output_image)

if __name__ == "__main__":
    main()
