import math
import os
from typing import Iterable, Tuple

import ezdxf
from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier, Arc, Path


def _sample_path(path: Path, step: float) -> Iterable[Tuple[float, float]]:
    length = path.length(error=1e-3)
    if length == 0:
        return []
    num = max(2, int(math.ceil(length / step)))
    for i in range(num + 1):
        t = i / num
        pt = path.point(t)
        yield (pt.real, pt.imag)


def _flip_y(points: Iterable[Tuple[float, float]]) -> Iterable[Tuple[float, float]]:
    for x, y in points:
        yield (x, -y)


def svg_to_dxf(svg_path: str, dxf_path: str, sample_step: float = 1.5, flip_y: bool = True, close_paths: bool = True) -> None:
    if not os.path.exists(svg_path):
        raise FileNotFoundError(svg_path)

    paths, attributes, svg_attributes = svg2paths2(svg_path)

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    for path in paths:
        sampled = list(_sample_path(path, step=sample_step))
        if flip_y:
            sampled = list(_flip_y(sampled))

        if len(sampled) >= 2:
            msp.add_lwpolyline(sampled, format="xy", close=close_paths)

    # Optional: set units to millimeters
    doc.units = ezdxf.units.MM
    doc.saveas(dxf_path)


