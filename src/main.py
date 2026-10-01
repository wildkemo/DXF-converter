import argparse
import os
import sys
import glob
import shutil

from src.config import VectorizationConfig, ThresholdMethod, CoordinateSpace
from src.vectorization import VectorizationPipeline
from src.serialization import to_json, to_svg, to_points_image
import ezdxf


import numpy as np

def save_dxf(result, output_path):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    coord_space = result.metadata.coordinate_space
    height = result.metadata.height
    
    # SVG/Images use screen coordinates (Y goes down). DXF uses Cartesian (Y goes up).
    # If the vectorization was done in pixel/normalized space, we must invert Y for DXF.
    invert_y = "cartesian" not in coord_space
    
    layers_created = set()
    
    for c in result.contours:
        if len(c.points) > 1:
            if invert_y:
                if "normalized" in coord_space:
                    pts = [(p.x, 1.0 - p.y, 0) for p in c.points]
                else:
                    pts = [(p.x, height - p.y, 0) for p in c.points]
            else:
                pts = [(p.x, p.y, 0) for p in c.points]
                
            # For closed curves, avoid duplicated first/last point in fit points (AutoCAD closes automatically)
            if c.is_closed and len(pts) > 2:
                p_first = np.array(pts[0])
                p_last = np.array(pts[-1])
                if np.linalg.norm(p_first - p_last) < 1e-4:
                    pts = pts[:-1]

            is_dup = getattr(c, 'is_duplicate', False)
            if is_dup:
                layer_name = "CNC_OFFSET"
                color = 4  # Cyan for CNC offset toolpaths
            else:
                is_hole = getattr(c, 'is_hole', False)
                level = getattr(c, 'hierarchy_level', 0)
                layer_name = f"LEVEL_{level}_{'HOLE' if is_hole else 'OUTER'}"
                color = 1 if is_hole else 7

            if layer_name not in layers_created:
                doc.layers.add(name=layer_name, color=color)
                layers_created.add(layer_name)

            spline = msp.add_spline(pts, dxfattribs={"layer": layer_name})
            if c.is_closed:
                spline.closed = True
    doc.saveas(output_path)


def main():
    parser = argparse.ArgumentParser(description="Image Contour Detection and Vectorization Engine")
    
    parser.add_argument("--input", type=str, default=None, help="Path to a single input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output directory for a single image")
    parser.add_argument("-d", "--dataset", default="dxf coloring/dataset", help="Path to dataset directory containing pair_n folders")
    parser.add_argument("-t", "--threshold-method", type=str, default="otsu", 
                        choices=[m.value for m in ThresholdMethod],
                        help="Binarization method")
    parser.add_argument("-e", "--epsilon-factor", type=float, default=0.005, help="RDP simplification factor")
    parser.add_argument("-c", "--coord-space", type=str, default="pixel",
                        choices=[cs.value for cs in CoordinateSpace],
                        help="Output coordinate space")
    parser.add_argument("--min-area", type=float, default=10.0, help="Minimum contour area to keep")
    parser.add_argument("--min-perimeter", type=float, default=10.0, help="Minimum contour perimeter to keep")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visualizations")
    parser.add_argument("--smooth", action="store_true", default=True, help="Enable contour smoothing (default: True)")
    parser.add_argument("--no-smooth", action="store_false", dest="smooth", help="Disable contour smoothing")
    parser.add_argument("--smooth-sigma", type=float, default=1.5, help="Contour smoothing Gaussian sigma (default: 1.5)")
    parser.add_argument("--preserve-corners", action="store_true", default=True, help="Preserve sharp corners during smoothing (default: True)")
    parser.add_argument("--no-preserve-corners", action="store_false", dest="preserve_corners", help="Disable corner preservation")
    parser.add_argument("--corner-threshold", type=float, default=50.0, help="Corner turning angle threshold in degrees (default: 50.0)")
    
    # CNC Line Duplication / Offset arguments
    parser.add_argument("-D", "--duplicate-distance", type=float, default=None,
                        help="CNC duplicate line offset distance (in coordinate units/pixels, e.g. 3.0 or -2.0 for cutter compensation)")
    parser.add_argument("--duplicate-both-sides", action="store_true",
                        help="Generate CNC offset lines on both sides (+distance and -distance)")
    parser.add_argument("--miter-limit", type=float, default=2.5,
                        help="Miter limit for sharp corners during line offsetting (default: 2.5)")
                        
    # Enhanced detection flag
    parser.add_argument("--detect-more", action="store_true",
                        help="Ultra-detailed detection mode: maximizes detected contours using multi-channel hybrid edges and micro-feature thresholds")
    
    args = parser.parse_args()
    
    # Setup config
    config = VectorizationConfig()
    config.threshold_method = ThresholdMethod(args.threshold_method)
    config.epsilon_factor = args.epsilon_factor
    config.coordinate_space = CoordinateSpace(args.coord_space)
    config.min_area = args.min_area
    config.min_perimeter = args.min_perimeter
    config.debug_visualization = not args.no_debug
    config.smooth_contours = args.smooth
    config.smooth_sigma = args.smooth_sigma
    config.corner_preservation = args.preserve_corners
    config.corner_threshold_deg = args.corner_threshold
    
    # CNC Offset configuration
    config.duplicate_distance = args.duplicate_distance
    config.duplicate_both_sides = args.duplicate_both_sides
    config.miter_limit = args.miter_limit
    
    # When user specifies threshold-method or detect-more, activate grayscale / enhanced preprocessing
    if args.detect_more:
        config.threshold_method = ThresholdMethod.HYBRID_ALL
        config.use_grayscale = True
        config.min_area = min(args.min_area, 2.0)
        config.min_perimeter = min(args.min_perimeter, 4.0)
        config.clahe_clip_limit = 3.0
    elif args.threshold_method != "otsu" or "--threshold-method" in sys.argv or "-t" in sys.argv:
        config.use_grayscale = True
    
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
