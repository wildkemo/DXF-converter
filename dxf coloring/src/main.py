import os
import glob
import argparse
import colorsys
import ezdxf
from ezdxf import path as ezdxf_path, bbox
import cv2
import numpy as np


# --- Core rendering function ---

def resolve_entity_aci(entity, doc):
    """Resolve a DXF entity's ACI color index (for grouping purposes)."""
    if entity.dxf.hasattr("color") and entity.dxf.color not in (0, 256):
        return entity.dxf.color
    layer = doc.layers.get(entity.dxf.layer)
    return layer.dxf.color if layer else 7


def generate_shape_colors(shapes, seed):
    """Generate diverse, realistic fill colors for a list of shapes.

    Groups shapes by ACI color. Each group gets a base hue, and
    individual shapes within a group get varied saturation/brightness
    to create natural-looking tonal diversity.
    """
    rng = np.random.RandomState(seed)

    # Identify unique ACI groups
    aci_groups = sorted(set(s["aci"] for s in shapes if s["closed"]))
    n_groups = max(len(aci_groups), 1)

    # Assign evenly-spaced base hues across the color wheel, then shuffle
    base_hues = [(i / n_groups + rng.uniform(0, 0.5 / n_groups)) % 1.0 for i in range(n_groups)]
    rng.shuffle(base_hues)
    group_hue = {aci: base_hues[i] for i, aci in enumerate(aci_groups)}

    colors = {}
    for s in shapes:
        idx = s["idx"]
        if not s["closed"]:
            colors[idx] = (0, 0, 0)  # Open splines: black stroke
            continue

        aci = s["aci"]
        h = group_hue.get(aci, rng.random())

        # Per-shape variation: slight hue shift + varied saturation & value
        h = (h + rng.uniform(-0.04, 0.04)) % 1.0
        sat = rng.uniform(0.35, 0.75)   # Moderate saturation (not neon)
        val = rng.uniform(0.55, 0.92)   # Medium-to-bright value (not washed out, not dark)

        # HSV → RGB
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        colors[idx] = (int(r * 255), int(g * 255), int(b * 255))

    return colors


def dxf_to_pixel(pts_2d, min_x, max_y, scale, padding):
    """Convert DXF world coordinates to pixel coordinates (Y-flipped)."""
    px = ((pts_2d[:, 0] - min_x) * scale + padding).astype(np.int32)
    py = ((max_y - pts_2d[:, 1]) * scale + padding).astype(np.int32)
    return np.column_stack([px, py])


def convert_dxf_to_image(input_path, output_path, target_size=2048, padding=10):
    """Render a DXF file to a colored JPEG image.

    1. Flatten every SPLINE entity into a polygon via ezdxf.path.
    2. Sort closed shapes by area (largest first) for painter's algorithm.
    3. Fill shapes with diverse realistic colors, then draw boundaries on top.
    """
    print(f"Processing: {input_path}")
    doc = ezdxf.readfile(input_path)
    msp = doc.modelspace()

    entities = list(msp)
    if not entities:
        print(f"  Skipping (empty modelspace): {input_path}")
        return

    # --- Bounding box & scaling ---
    ext = bbox.extents(msp, fast=True)
    if not ext.has_data:
        print(f"  Skipping (no bounding box): {input_path}")
        return

    min_x, min_y = ext.extmin.x, ext.extmin.y
    max_x, max_y = ext.extmax.x, ext.extmax.y
    dxf_w = max_x - min_x
    dxf_h = max_y - min_y

    if dxf_w < 1e-6 or dxf_h < 1e-6:
        print(f"  Skipping (degenerate bounds): {input_path}")
        return

    scale = target_size / max(dxf_w, dxf_h)
    img_w = int(dxf_w * scale) + 2 * padding
    img_h = int(dxf_h * scale) + 2 * padding

    # --- Flatten all entities ---
    shapes = []
    for i, entity in enumerate(entities):
        try:
            p = ezdxf_path.make_path(entity)
        except Exception:
            continue

        pts = list(p.flattening(0.1))
        if len(pts) < 2:
            continue

        pts_2d = np.array([(pt.x, pt.y) for pt in pts])

        # Signed area via shoelace formula
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        pixel_pts = dxf_to_pixel(pts_2d, min_x, max_y, scale, padding)

        is_closed = False
        if hasattr(entity, "closed"):
            is_closed = entity.closed

        shapes.append({
            "idx": i,
            "pixel_pts": pixel_pts,
            "area": area,
            "closed": is_closed,
            "aci": resolve_entity_aci(entity, doc),
        })

    if not shapes:
        print(f"  Skipping (no valid shapes): {input_path}")
        return

    # --- Generate diverse, realistic colors ---
    seed = hash(os.path.basename(input_path)) % (2**31)
    colors = generate_shape_colors(shapes, seed)

    # --- Render: painter's algorithm (largest area first) ---
    shapes.sort(key=lambda s: s["area"], reverse=True)

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # 1. Fill closed shapes
    for s in shapes:
        if s["closed"] and s["area"] > 1:
            r, g, b = colors[s["idx"]]
            cv2.fillPoly(img, [s["pixel_pts"]], (b, g, r))

    # 2. Draw boundaries on top
    for s in shapes:
        if s["closed"]:
            cv2.polylines(img, [s["pixel_pts"]], True, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [s["pixel_pts"]], False, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {output_path} ({img_w}x{img_h}, {len(shapes)} shapes)")


# --- Batch processing ---

def main():
    parser = argparse.ArgumentParser(description="Render DXF files to colored JPEG images")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataset = os.path.join(base_dir, "dataset")

    parser.add_argument(
        "--dataset", type=str, default=default_dataset,
        help="Path to dataset directory containing pair_* folders with perfect.dxf",
    )
    parser.add_argument(
        "--target-size", type=int, default=2048,
        help="Target size in pixels for the longest image dimension (default: 2048)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum number of pairs to process (0 for unlimited)",
    )

    args = parser.parse_args()
    dataset_dir = os.path.abspath(args.dataset)

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return

    pair_folders = sorted(glob.glob(os.path.join(dataset_dir, "pair_*")))
    if not pair_folders:
        print(f"No pair_* folders found in {dataset_dir}")
        return

    if args.limit > 0:
        pair_folders = pair_folders[: args.limit]

    success = 0
    skipped = 0
    for pair_folder in pair_folders:
        dxf_path = os.path.join(pair_folder, "perfect.dxf")
        if not os.path.isfile(dxf_path):
            skipped += 1
            continue

        output_path = os.path.join(pair_folder, "image.jpeg")
        try:
            convert_dxf_to_image(dxf_path, output_path, target_size=args.target_size)
            success += 1
        except Exception as e:
            print(f"  Error processing {dxf_path}: {e}")
            skipped += 1

    print(f"\nDone. Processed {success} pairs, skipped {skipped}.")


if __name__ == "__main__":
    main()
