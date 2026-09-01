import ezdxf
import os
import json
from src.dxf_parser import parse_dxf

def test_dxf_parser(tmp_path):
    # 1. Create a dummy DXF file
    dxf_path = str(tmp_path / "test_input.dxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    doc.layers.add(name="MY_LAYER", color=3)
    msp.add_circle((0, 0), radius=10, dxfattribs={"layer": "MY_LAYER"})
    
    if "DXF_CONVERTER" not in doc.appids:
        doc.appids.add("DXF_CONVERTER")
    line = msp.add_line((0, 0), (10, 10))
    line.set_xdata("DXF_CONVERTER", [(1000, "TEST_STRING")])
    
    doc.saveas(dxf_path)
    
    # 2. Run the parser
    output_dir = str(tmp_path / "custom")
    parse_dxf(dxf_path, output_dir)
    
    # 3. Verify output
    json_path = os.path.join(output_dir, "test_input_parsed.json")
    assert os.path.exists(json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    assert "header" in data
    assert "layers" in data
    assert "modelspace" in data
    
    # Check layers
    layer_names = [l['name'] for l in data['layers']]
    assert "MY_LAYER" in layer_names
    
    # Check entities
    assert len(data['modelspace']) == 2
    
    types = [e['dxftype'] for e in data['modelspace']]
    assert 'CIRCLE' in types
    assert 'LINE' in types
    
    # Check specific entity extraction
    for e in data['modelspace']:
        if e['dxftype'] == 'CIRCLE':
            assert e['radius'] == 10.0
            assert e['center'] == [0.0, 0.0, 0.0]
        elif e['dxftype'] == 'LINE':
            assert 'xdata' in e
            assert 'DXF_CONVERTER' in e['xdata']
            assert e['xdata']['DXF_CONVERTER'][1][1] == "TEST_STRING"
