import ezdxf

doc = ezdxf.new()
msp = doc.modelspace()
msp.add_lwpolyline([(0,0), (10,0), (10,10)], close=True, dxfattribs={'layer': 'TEST'})
doc.saveas("test.dxf")

def parse_entity(entity):
    data = {
        'dxftype': entity.dxftype(),
        'layer': entity.dxf.layer,
        'color': entity.dxf.color,
    }
    # Generic dump of all DXF attributes to "neglect nothing"
    data['attribs'] = {k: str(v) for k, v in entity.dxf.all_existing_dxf_attribs().items()}
    
    # Extract geometry points if applicable
    if hasattr(entity, 'get_points'):
        try:
            data['points'] = [list(p) for p in entity.get_points()]
        except:
            pass
    elif hasattr(entity, 'points'):
        try:
            data['points'] = [list(p) for p in entity.points]
        except:
            pass
            
    if hasattr(entity, 'vertices'):
        try:
            data['vertices'] = [list(v.dxf.location) for v in entity.vertices]
        except:
            pass
            
    # Extract XDATA to neglect nothing
    xdata_dict = {}
    if entity.xdata:
        for appid, tags in entity.xdata.items():
            xdata_dict[appid] = [(t.code, t.value) for t in tags]
    if xdata_dict:
        data['xdata'] = xdata_dict
        
    return data

doc2 = ezdxf.readfile("test.dxf")
entities = []
for entity in doc2.modelspace():
    entities.append(parse_entity(entity))
    
print(f"Parsed {len(entities)} entities.")
import json
print(json.dumps(entities[0], indent=2))
