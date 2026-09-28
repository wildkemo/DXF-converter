"""Inference pipeline: DXF → render → U-Net → threshold → re-vectorize → DXF."""

import os
import argparse
import numpy as np
import cv2
import torch

from .model import UNet
from .render_dxf import render_dxf_to_image


def infer_and_correct(input_dxf, output_dxf, model_path, image_size=512):
    """Run the full correction pipeline on a single DXF file.

    Steps:
        1. Render the input DXF to a binary raster image.
        2. Pass through the U-Net to produce a clean image.
        3. Threshold and re-vectorize using OpenCV contour detection.
        4. Write the clean contours as splines to a new DXF file.

    Args:
        input_dxf:  Path to the corrupted / noisy DXF.
        output_dxf: Where to save the corrected DXF.
        model_path: Path to the trained ``best_unet.pt`` checkpoint.
        image_size: Must match the resolution used during training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load model ──
    model = UNet(in_channels=1, out_channels=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: model not found at {model_path}")
        return
    model.eval()

    # ── Render input DXF ──
    print(f"Rendering {input_dxf} to {image_size}×{image_size} image...")
    input_img = render_dxf_to_image(input_dxf, size=image_size)
    ink_pixels = (input_img > 0.5).sum()
    print(f"  Ink pixels: {ink_pixels} / {image_size * image_size} "
          f"({100 * ink_pixels / image_size**2:.1f}%)")

    # ── U-Net forward pass ──
    x = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    print(f"  Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")

    # ── Threshold to binary ──
    binary = (pred > 0.5).astype(np.uint8) * 255

    clean_pixels = (binary > 0).sum()
    print(f"  Clean ink pixels: {clean_pixels} "
          f"({100 * clean_pixels / image_size**2:.1f}%)")

    # ── Re-vectorize via contour detection ──
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Found {len(contours)} contours in cleaned image")

    # ── Write to DXF ──
    import ezdxf
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    spline_count = 0
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue

        # Convert pixel coords to float, flip Y back to DXF convention
        fit_points = [(float(p[0]), float(image_size - 1 - p[1]))
                      for p in pts]
        spline = msp.add_spline(fit_points=fit_points)

        # Close if the contour is roughly closed
        if np.linalg.norm(pts[0] - pts[-1]) < 5:
            spline.closed = True
        spline_count += 1

    doc.saveas(output_dxf)
    print(f"Saved corrected DXF to {output_dxf} ({spline_count} splines)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="U-Net DXF Correction Inference")
    parser.add_argument("--input", required=True,
                        help="Path to corrupted DXF")
    parser.add_argument("--output", required=True,
                        help="Path to save corrected DXF")
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint (best_unet.pt)")
    parser.add_argument("--size", type=int, default=512,
                        help="Image size (must match training)")
    args = parser.parse_args()

    infer_and_correct(args.input, args.output, args.model, args.size)
