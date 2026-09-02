import argparse
import os
import sys
import glob
import shutil

from src.config import VectorizationConfig, ThresholdMethod, CoordinateSpace
from src.vectorization import VectorizationPipeline
from src.serialization import to_json, to_svg
import ezdxf


def save_dxf(result, output_path):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    for c in result.contours:
        if len(c.points) > 1:
            pts = [(p.x, p.y) for p in c.points]
            msp.add_lwpolyline(pts, close=c.is_closed)
    doc.saveas(output_path)


def main():
    parser = argparse.ArgumentParser(description="Image Contour Detection and Vectorization Engine")
    
    parser.add_argument("-d", "--dataset", default="dxf coloring/dataset", help="Path to dataset directory containing pair_n folders")
    parser.add_argument("-t", "--threshold-method", type=str, default="otsu", 
                        choices=["otsu", "adaptive_gaussian", "adaptive_mean", "canny", "binary_fixed"],
                        help="Binarization method")
    parser.add_argument("-e", "--epsilon-factor", type=float, default=0.005, help="RDP simplification factor")
    parser.add_argument("-c", "--coord-space", type=str, default="pixel",
                        choices=["pixel", "normalized", "cartesian_pixel", "cartesian_normalized"],
                        help="Output coordinate space")
    parser.add_argument("--min-area", type=float, default=10.0, help="Minimum contour area to keep")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visualizations")
    
    args = parser.parse_args()
    
    dataset_dir = args.dataset
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)
        
    # Setup config
    config = VectorizationConfig()
    config.threshold_method = ThresholdMethod(args.threshold_method)
    config.epsilon_factor = args.epsilon_factor
    config.coordinate_space = CoordinateSpace(args.coord_space)
    config.min_area = args.min_area
    config.debug_visualization = not args.no_debug
    
    pipeline = VectorizationPipeline(config)
    
    # Find all pair folders
    pair_folders = glob.glob(os.path.join(dataset_dir, "pair_*"))
    if not pair_folders:
        print(f"No pair folders found in {dataset_dir}")
        sys.exit(1)
        
    # Process each pair
    for pair_folder in pair_folders:
        if not os.path.isdir(pair_folder):
            continue
            
        input_image = os.path.join(pair_folder, "image.png")
        if not os.path.exists(input_image):
            print(f"Warning: No image.png found in {pair_folder}")
            continue
            
        print(f"Processing '{input_image}'...")
        
        try:
            result = pipeline.process_image(input_image, pair_folder)
            
            if result.metrics.total_contours < 100:
                print(f"Fewer than 100 contours ({result.metrics.total_contours}) found. Removing {pair_folder}...")
                shutil.rmtree(pair_folder)
                continue
            
            # Save DXF
            dxf_path = os.path.join(pair_folder, "not perfict.dxf")
            save_dxf(result, dxf_path)
            
            # Save JSON
            json_path = os.path.join(pair_folder, "contours.json")
            to_json(result, output_path=json_path)
            
            # Save SVG
            svg_path = os.path.join(pair_folder, "contours.svg")
            to_svg(result, output_path=svg_path)
            
            print(f"Vectorization complete for {pair_folder}: {result.metrics.total_contours} contours extracted.")
            print(f"Results saved to '{pair_folder}'")
            
        except Exception as e:
            print(f"Error during processing {pair_folder}: {str(e)}")

if __name__ == "__main__":
    main()
