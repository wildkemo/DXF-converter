import argparse
import sys
import os
import json
import ezdxf

def parse_entity(entity):
    """Deeply inspect a DXF entity and convert it to a dictionary."""
    ent_data = {
        'dxftype': entity.dxftype()
    }
    
    # Generic dump of all DXF attributes (handle, layer, color, etc.) to neglect nothing
    ent_data['attribs'] = {k: str(v) for k, v in entity.dxf.all_existing_dxf_attribs().items()}
    
    # Extract points for Polylines
    if hasattr(entity, 'get_points'):
        try: ent_data['points'] = [list(p) for p in entity.get_points()]
        except: pass
    elif hasattr(entity, 'points'):
        try: ent_data['points'] = [list(p) for p in entity.points]
        except: pass
    elif hasattr(entity, 'vertices'):
        try: ent_data['vertices'] = [list(v.dxf.location) for v in entity.vertices]
        except: pass
        
    # Extract explicit geometry for standard shapes
    if entity.dxftype() == 'LINE':
        try:
            ent_data['start'] = [entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z]
            ent_data['end'] = [entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z]
        except: pass
    elif entity.dxftype() == 'CIRCLE':
        try:
            ent_data['center'] = [entity.dxf.center.x, entity.dxf.center.y, entity.dxf.center.z]
            ent_data['radius'] = entity.dxf.radius
        except: pass
    elif entity.dxftype() == 'ARC':
        try:
            ent_data['center'] = [entity.dxf.center.x, entity.dxf.center.y, entity.dxf.center.z]
            ent_data['radius'] = entity.dxf.radius
            ent_data['start_angle'] = entity.dxf.start_angle
            ent_data['end_angle'] = entity.dxf.end_angle
        except: pass
    elif entity.dxftype() in ('TEXT', 'MTEXT'):
        try:
            ent_data['text'] = entity.text
            if hasattr(entity.dxf, 'insert'):
                ent_data['insert'] = [entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z]
        except: pass
    elif entity.dxftype() == 'SPLINE':
        try:
            ent_data['fit_points'] = [list(p) for p in entity.fit_points]
            ent_data['control_points'] = [list(p) for p in entity.control_points]
            ent_data['degree'] = entity.dxf.degree
            ent_data['closed'] = entity.closed
            ent_data['knots'] = list(entity.knots)
        except: pass
        
    # Extract XDATA (Extended Data) to ensure metadata is preserved
    xdata_dict = {}
    if hasattr(entity, 'xdata') and entity.xdata and hasattr(entity.xdata, 'data'):
        for appid, tags in entity.xdata.data.items():
            xdata_dict[appid] = [(t.code, str(t.value)) for t in tags]
    if xdata_dict:
        ent_data['xdata'] = xdata_dict
        
    return ent_data

def parse_dxf(input_path: str, output_dir: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
        
    print(f"Reading DXF file: {input_path}...")
    try:
        doc = ezdxf.readfile(input_path)
    except Exception as e:
        print(f"Error reading DXF: {e}")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        'header': {},
        'layers': [],
        'blocks': {},
        'modelspace': []
    }
    
    print("Parsing Headers...")
    for k in doc.header.varnames():
        data['header'][k] = str(doc.header[k])
        
    print("Parsing Layers...")
    for layer in doc.layers:
        data['layers'].append({
            'name': layer.dxf.name,
            'color': layer.dxf.color,
            'linetype': layer.dxf.linetype
        })
        
    print("Parsing Modelspace...")
    for entity in doc.modelspace():
        data['modelspace'].append(parse_entity(entity))
        
    print("Parsing Blocks...")
    for block in doc.blocks:
        if block.name.startswith('*'): continue # Skip internal anonymous blocks
        blk_data = []
        for entity in block:
            blk_data.append(parse_entity(entity))
        data['blocks'][block.name] = blk_data
        
    filename = os.path.basename(input_path)
    base_name, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{base_name}_parsed.json")
    
    print(f"Writing parsed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Standalone DXF to JSON Parser")
    parser.add_argument("input", help="Input DXF file")
    
    args = parser.parse_args()
    # Always save to a 'custom' folder in the root of the project
    output_dir = os.path.join(os.getcwd(), "custom")
    
    parse_dxf(args.input, output_dir)

if __name__ == "__main__":
    main()
