import argparse
import os
import sys
import glob
import shutil

from src.config import VectorizationConfig, ThresholdMethod, CoordinateSpace
from src.vectorization import VectorizationPipeline
from src.serialization import to_json, to_svg, to_points_image
import ezdxf


def save_dxf(result, output_path):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    coord_space = result.metadata.coordinate_space
    height = result.metadata.height
    
    # SVG/Images use screen coordinates (Y goes down). DXF uses Cartesian (Y goes up).
    # If the vectorization was done in pixel/normalized space, we must invert Y for DXF.
    invert_y = "cartesian" not in coord_space
    
    for c in result.contours:
        if len(c.points) > 1:
            if invert_y:
                if "normalized" in coord_space:
                    pts = [(p.x, 1.0 - p.y, 0) for p in c.points]
                else:
                    pts = [(p.x, height - p.y, 0) for p in c.points]
            else:
                pts = [(p.x, p.y, 0) for p in c.points]
                
            spline = msp.add_spline(pts)
            if c.is_closed:
                spline.closed = True
    doc.saveas(output_path)


def main():
    parser = argparse.ArgumentParser(description="Image Contour Detection and Vectorization Engine")
    
    parser.add_argument("--input", type=str, default=None, help="Path to a single input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output directory for a single image")
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
    
    # Setup config
    config = VectorizationConfig()
    config.threshold_method = ThresholdMethod(args.threshold_method)
    config.epsilon_factor = args.epsilon_factor
    config.coordinate_space = CoordinateSpace(args.coord_space)
    config.min_area = args.min_area
    config.debug_visualization = not args.no_debug
    
    pipeline = VectorizationPipeline(config)
    
    # Single file processing mode
    if args.input and args.output:
        if not os.path.exists(args.input):
            print(f"Error: Input image '{args.input}' does not exist.")
            sys.exit(1)
            
        os.makedirs(args.output, exist_ok=True)
        print(f"Processing single image '{args.input}'...")
        
        try:
            result = pipeline.process_image(args.input, args.output)
            
            # Save DXF
            dxf_path = os.path.join(args.output, "output.dxf")
            save_dxf(result, dxf_path)
            
            # Save JSON
            json_path = os.path.join(args.output, "contours.json")
            to_json(result, output_path=json_path)
            
            # Save SVG
            svg_path = os.path.join(args.output, "contours.svg")
            to_svg(result, output_path=svg_path)
            
            # Save points image
            img_path = os.path.join(args.output, "points.png")
            to_points_image(result, output_path=img_path)
            
            print(f"Vectorization complete: {result.metrics.total_contours} contours extracted.")
            print(f"Results saved to '{args.output}'")
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            
        return

    # Dataset processing mode
    dataset_dir = args.dataset
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)
        
    # Find all pair folders
    pair_folders = glob.glob(os.path.join(dataset_dir, "pair_*"))
    if not pair_folders:
        print(f"No pair folders found in {dataset_dir}")
        sys.exit(1)
        
    # Process each pair
    for pair_folder in pair_folders:
        if not os.path.isdir(pair_folder):
            continue
            
        input_image = os.path.join(pair_folder, "image.jpeg")
        if not os.path.exists(input_image):
            print(f"Warning: No image.jpeg found in {pair_folder}")
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
            
            # Save points image
            img_path = os.path.join(pair_folder, "points.png")
            to_points_image(result, output_path=img_path)
            
            print(f"Vectorization complete for {pair_folder}: {result.metrics.total_contours} contours extracted.")
            print(f"Results saved to '{pair_folder}'")
            
        except Exception as e:
            print(f"Error during processing {pair_folder}: {str(e)}")

if __name__ == "__main__":
    main()
