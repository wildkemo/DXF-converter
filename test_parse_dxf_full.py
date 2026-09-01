import ezdxf
import json
import os

doc = ezdxf.readfile("test.dxf")

data = {
    'header': {},
    'layers': [],
    'blocks': {},
    'modelspace': []
}

for k in doc.header.varnames():
    data['header'][k] = str(doc.header[k])
    
for layer in doc.layers:
    data['layers'].append({
        'name': layer.dxf.name,
        'color': layer.dxf.color,
        'linetype': layer.dxf.linetype
    })
    
def parse_entity(entity):
    ent_data = {
        'dxftype': entity.dxftype()
    }
    ent_data['attribs'] = {k: str(v) for k, v in entity.dxf.all_existing_dxf_attribs().items()}
    
    if hasattr(entity, 'get_points'):
        try:
            ent_data['points'] = [list(p) for p in entity.get_points()]
        except: pass
    elif hasattr(entity, 'points'):
        try:
            ent_data['points'] = [list(p) for p in entity.points]
        except: pass
    elif hasattr(entity, 'vertices'):
        try:
            ent_data['vertices'] = [list(v.dxf.location) for v in entity.vertices]
        except: pass
        
    # Handling specific geometry data that might not be in points
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
        
    xdata_dict = {}
    if entity.xdata:
        for appid, tags in entity.xdata.items():
            xdata_dict[appid] = [(t.code, t.value) for t in tags]
    if xdata_dict:
        ent_data['xdata'] = xdata_dict
        
    return ent_data

for entity in doc.modelspace():
    data['modelspace'].append(parse_entity(entity))
    
for block in doc.blocks:
    if block.name.startswith('*'): continue # skip anonymous blocks
    blk_data = []
    for entity in block:
        blk_data.append(parse_entity(entity))
    data['blocks'][block.name] = blk_data
    
print(f"Header vars: {len(data['header'])}")
print(f"Layers: {len(data['layers'])}")
print(f"Modelspace entities: {len(data['modelspace'])}")
