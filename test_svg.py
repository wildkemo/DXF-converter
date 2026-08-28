def to_svg(result, output_path: str):
    w, h = result.metadata.width, result.metadata.height
    coord_space = result.metadata.coordinate_space
    
    if "normalized" in coord_space:
        viewbox = "0 0 1 1"
        stroke_width = 0.001
    else:
        viewbox = f"0 0 {w} {h}"
        stroke_width = 1.0
        
    if "cartesian" in coord_space:
        if "normalized" in coord_space:
            transform = 'transform="translate(0, 1) scale(1, -1)"'
        else:
            transform = f'transform="translate(0, {h}) scale(1, -1)"'
    else:
        transform = ""

    svg = [
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg width="{w}" height="{h}" viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">',
        f'  <g fill="none" stroke="black" stroke-width="{stroke_width}" {transform}>'
    ]
    
    for c in result.contours:
        pts_str = " ".join([f"{p.x},{p.y}" for p in c.points])
        tag = "polygon" if c.is_closed else "polyline"
        svg.append(f'    <{tag} points="{pts_str}" />')
        
    svg.append('  </g>')
    svg.append('</svg>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg))
