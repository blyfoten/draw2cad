"""Read any DXF back as categorised linework in mm (for scoring)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf import path as ezpath

from draw2cad.layers import Category, category_of_layer

INSUNITS_TO_MM = {0: 1.0, 4: 1.0, 5: 10.0, 6: 1000.0, 1: 25.4, 2: 304.8}


# --- reading ---------------------------------------------------------------------------------------


@dataclass
class Linework:
    """Polylines (N x 2 arrays, mm) per category. Blocks are exploded into their parts."""

    lines: dict[Category, list[np.ndarray]] = field(default_factory=dict)

    def points(self, cats, step: float = 10.0) -> np.ndarray:
        pts = [densify(pl, step) for c in cats for pl in self.lines.get(c, [])]
        return np.vstack(pts) if pts else np.zeros((0, 2))


def densify(pl: np.ndarray, step: float) -> np.ndarray:
    if len(pl) < 2:
        return pl
    out = [pl[:1]]
    for p, q in zip(pl[:-1], pl[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(q - p) / step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(p + (q - p) * t)
    return np.vstack(out)


def _entity_polylines(e) -> list[np.ndarray]:
    try:
        if e.dxftype() == "HATCH":
            paths = list(ezpath.from_hatch(e))
        else:
            paths = [ezpath.make_path(e)]
    except (TypeError, ValueError, ezdxf.DXFError):
        return []
    out = []
    for p in paths:
        for sub in p.sub_paths():
            pts = np.array([(v.x, v.y) for v in sub.flattening(distance=1.0)])
            if len(pts) >= 2:
                out.append(pts)
    return out


SKIP = {"TEXT", "MTEXT", "DIMENSION", "ATTRIB", "ATTDEF", "POINT", "VIEWPORT", "IMAGE"}


def read_dxf(path: Path) -> Linework:
    doc = ezdxf.readfile(path)
    scale = INSUNITS_TO_MM.get(doc.header.get("$INSUNITS", 0), 1.0)
    lw = Linework()

    def visit(e, inherited: str | None, depth: int = 0):
        t = e.dxftype()
        if t in SKIP:
            return
        # Entities on layer 0 inside a block take the INSERT's layer, as in CAD.
        layer = e.dxf.layer if e.dxf.layer != "0" or inherited is None else inherited
        if t == "INSERT":
            if depth <= 8:
                for v in e.virtual_entities():
                    visit(v, layer, depth + 1)
            return
        cat = category_of_layer(layer)
        for pl in _entity_polylines(e):
            lw.lines.setdefault(cat, []).append(pl * scale)

    for e in doc.modelspace():
        visit(e, None)
    return lw
