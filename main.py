import argparse
import os
import sys
import tempfile
from typing import List, Optional

from core.preprocessing import preprocess_to_pbm
from core.potrace_wrapper import potrace_to_svg, ensure_potrace_available
from core.dxf_writer import svg_to_dxf


def convert_image(input_path: str, output_dir: str, threshold: int, sample_step: float, flip_y: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    dxf_path = os.path.join(output_dir, f"{base_name}.dxf")

    with tempfile.TemporaryDirectory(prefix="img2dxf_") as tmp:
        pbm_path = preprocess_to_pbm(input_path, tmp, threshold=threshold)
        svg_path = os.path.join(tmp, f"{base_name}.svg")
        potrace_to_svg(pbm_path, svg_path)
        svg_to_dxf(svg_path, dxf_path, sample_step=sample_step, flip_y=flip_y)

    return dxf_path


def collect_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f.lower())[1] in supported
        ]
    return [input_path]


def run_cli(args: argparse.Namespace) -> None:
    ensure_potrace_available()
    images = collect_images(args.input)
    if not images:
        print("No images found.")
        sys.exit(1)

    print(f"Converting {len(images)} file(s) → {args.output}")
    for img in images:
        try:
            out_path = convert_image(
                img,
                args.output,
                threshold=args.threshold,
                sample_step=args.sample_step,
                flip_y=args.flip_y,
            )
            print(f"✔ {os.path.basename(img)} → {out_path}")
        except Exception as exc:
            print(f"✖ Failed for {img}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raster images (B/W) to DXF via Potrace and ezdxf",
    )
    parser.add_argument(
        "--input",
        help="Path to an image file or a folder of images (not required with --gui)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory to write DXF files",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binary threshold (0–255) for preprocessing",
    )
    parser.add_argument(
        "--sample-step",
        type=float,
        default=1.5,
        help="Polyline sampling step in SVG units (smaller = more points)",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip Y-axis to match typical CAD coordinate system",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch simple GUI instead of CLI",
    )
    return parser


def launch_gui():
    # Lazy import to keep CLI fast without Qt overhead
    from gui.main_window import launch

    launch()


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gui:
        launch_gui()
    else:
        if not args.input:
            parser.error("--input is required unless --gui is used")
        run_cli(args)


