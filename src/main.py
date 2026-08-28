import argparse
import os
import sys

from src.config import VectorizationConfig, ThresholdMethod, CoordinateSpace
from src.vectorization import VectorizationPipeline
from src.serialization import to_json, to_svg


def main():
    parser = argparse.ArgumentParser(description="Image Contour Detection and Vectorization Engine")
    
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
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
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
        
    os.makedirs(args.output, exist_ok=True)
    
    # Setup config
    config = VectorizationConfig()
    config.threshold_method = ThresholdMethod(args.threshold_method)
    config.epsilon_factor = args.epsilon_factor
    config.coordinate_space = CoordinateSpace(args.coord_space)
    config.min_area = args.min_area
    config.debug_visualization = not args.no_debug
    
    # Process
    print(f"Processing '{args.input}'...")
    pipeline = VectorizationPipeline(config)
    
    try:
        result = pipeline.process_image(args.input, args.output)
        
        # Save JSON
        json_path = os.path.join(args.output, "contours.json")
        to_json(result, output_path=json_path)
        
        # Save SVG
        svg_path = os.path.join(args.output, "contours.svg")
        to_svg(result, output_path=svg_path)
        
        print(f"Vectorization complete: {result.metrics.total_contours} contours extracted.")
        print(f"Vertex reduction: {result.metrics.vertex_reduction_pct:.2f}%")
        print(f"Results saved to '{args.output}'")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
