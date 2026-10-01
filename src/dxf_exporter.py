import json
import argparse
import sys
import os
import ezdxf

def export_dxf(json_path: str, output_path: str):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        sys.exit(1)
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Initialize AppID for Extended Data
    if "DXF_CONVERTER" not in doc.appids:
        doc.appids.add("DXF_CONVERTER")
        
    metadata = data.get('metadata', {})
    height = metadata.get('height', 0)
    coord_space = metadata.get('coordinate_space', 'pixel')
    
    # SVG/Pixel coordinates have Y pointing down. DXF expects Y pointing up.
    flip_y = coord_space in ['pixel', 'normalized']
    
    contours = data.get('contours', [])
    layers_created = set()
    
    for c in contours:
        pts = c.get('points', [])
        if not pts:
            continue
            
        is_hole = c.get('is_hole', False)
        level = c.get('hierarchy_level', 0)
        is_dup = c.get('is_duplicate', False)
        offset_dist = c.get('offset_distance', None)
        
        # Organize layers: CNC duplicates get dedicated CNC_OFFSET layer with Cyan color
        if is_dup:
            layer_name = "CNC_OFFSET"
            color = 4  # Cyan for CNC offset toolpaths
        else:
            layer_name = f"LEVEL_{level}_{'HOLE' if is_hole else 'OUTER'}"
            color = 1 if is_hole else 7  # Red for holes, Black/White for outers
            
        if layer_name not in layers_created:
            doc.layers.add(name=layer_name, color=color)
            layers_created.add(layer_name)
            
        # Map Coordinates
        dxf_points = []
        for p in pts:
            x, y = p['x'], p['y']
            if flip_y:
                y = height - y
            dxf_points.append((x, y, 0))
            
        # Create Spline
        is_closed = c.get('is_closed', True)
        if is_closed and len(dxf_points) > 2:
            p0 = dxf_points[0]
            pn = dxf_points[-1]
            dist_sq = (p0[0] - pn[0])**2 + (p0[1] - pn[1])**2
            if dist_sq < 1e-8:
                dxf_points = dxf_points[:-1]

        spline = msp.add_spline(dxf_points, dxfattribs={"layer": layer_name})
        if is_closed:
            spline.closed = True
        
        # Inject JSON metrics directly into CAD XDATA
        xdata = [
            (1000, f"ID:{c.get('id', -1)}"),
            (1000, f"PARENT:{c.get('parent_id') if c.get('parent_id') is not None else 'NONE'}"),
            (1040, c.get('area', 0.0)),
            (1040, c.get('perimeter', 0.0)),
            (1000, f"DUPLICATE:{is_dup}"),
            (1040, float(offset_dist) if offset_dist is not None else 0.0)
        ]
        spline.set_xdata("DXF_CONVERTER", xdata)
        
    doc.saveas(output_path)
    print(f"Successfully generated DXF: {output_path} ({len(contours)} geometries)")

def main():
    parser = argparse.ArgumentParser(description="JSON to DXF Exporter")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file (e.g., output/contours.json)")
    parser.add_argument("-o", "--output", required=True, help="Output DXF file (e.g., output/drawing.dxf)")
    
    args = parser.parse_args()
    export_dxf(args.input, args.output)

if __name__ == "__main__":
    main()
