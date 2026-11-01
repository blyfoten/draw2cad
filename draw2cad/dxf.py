"""Write a Plan (mm) to a layered DXF."""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from draw2cad.layers import DEFAULT_LAYERS, Category
from draw2cad.plan import Opening, Plan, Pt, Wall

# Dimension style for a mm drawing plotted at 1:100: 2.5 mm text, architectural ticks, whole mm.
DIM_OVERRIDE = {"dimtxt": 250, "dimtsz": 120, "dimexe": 120, "dimexo": 60, "dimgap": 60,
                "dimdec": 0, "dimtad": 1, "dimlfac": 1.0, "dimpost": "<>"}


def door_symbol(wall: Wall, o: Opening) -> tuple[Pt, Pt, Pt]:
    """Hinge point, open leaf end and closed leaf end for a door drawn open at 90 degrees."""
    ux, uy = wall.direction
    nx, ny = -uy * o.swing, ux * o.swing
    cx, cy = o.center(wall)
    s = -1 if o.hinge_at_start else 1
    jx, jy = cx + s * ux * o.width / 2, cy + s * uy * o.width / 2
    hinge = (jx + nx * wall.thickness / 2, jy + ny * wall.thickness / 2)
    leaf = (hinge[0] + nx * o.width, hinge[1] + ny * o.width)
    closed = (hinge[0] - s * ux * o.width, hinge[1] - s * uy * o.width)
    return hinge, leaf, closed


def window_symbol(wall: Wall, o: Opening) -> list[tuple[Pt, Pt]]:
    """Jamb lines plus frame and glass lines along the wall."""
    ux, uy = wall.direction
    nx, ny = -uy, ux
    cx, cy = o.center(wall)
    h, t = o.width / 2, wall.thickness / 2
    segs = []
    for k in (-1, 1):  # jambs
        px, py = cx + k * ux * h, cy + k * uy * h
        segs.append(((px - nx * t, py - ny * t), (px + nx * t, py + ny * t)))
    for f in (-1.0, -0.15, 0.15, 1.0):  # outer face, glass pair, inner face
        px, py = cx + nx * t * f, cy + ny * t * f
        segs.append(((px - ux * h, py - uy * h), (px + ux * h, py + uy * h)))
    return segs


def iter_polygons(geom):
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms
    else:
        for g in getattr(geom, "geoms", []):
            yield from iter_polygons(g)


def new_doc() -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4  # mm
    doc.header["$MEASUREMENT"] = 1  # metric
    colors = {Category.WALL_EXT: 4, Category.WALL_INT: 3, Category.WALL: 3, Category.DOOR: 2,
              Category.WINDOW: 6, Category.DIMENSION: 1, Category.TEXT: 7, Category.ROOM: 5, Category.OTHER: 8}
    for cat, name in DEFAULT_LAYERS.items():
        doc.layers.add(name, color=colors.get(cat, 7))
    return doc


def plan_to_dxf(plan: Plan, out: Path) -> None:
    doc = new_doc()
    msp = doc.modelspace()
    L = {cat: {"layer": name} for cat, name in DEFAULT_LAYERS.items()}

    cuts = unary_union([o.cut(plan.wall(o.wall)) for o in plan.openings]) if plan.openings else None
    ext = unary_union([w.polygon() for w in plan.walls if w.exterior])
    inner = unary_union([w.polygon() for w in plan.walls if not w.exterior]).difference(ext)
    for solid, cat in ((ext, Category.WALL_EXT), (inner, Category.WALL_INT)):
        if cuts is not None:
            solid = solid.difference(cuts)
        for poly in iter_polygons(solid):
            for ring in (poly.exterior, *poly.interiors):
                msp.add_lwpolyline(list(ring.coords)[:-1], close=True, dxfattribs=L[cat])

    for o in plan.openings:
        wall = plan.wall(o.wall)
        if o.kind == "door":
            hinge, leaf, closed = door_symbol(wall, o)
            msp.add_line(hinge, leaf, dxfattribs=L[Category.DOOR])
            a0 = math.degrees(math.atan2(leaf[1] - hinge[1], leaf[0] - hinge[0]))
            a1 = math.degrees(math.atan2(closed[1] - hinge[1], closed[0] - hinge[0]))
            if (a1 - a0) % 360 > 180:
                a0, a1 = a1, a0
            msp.add_arc(hinge, o.width, a0, a1, dxfattribs=L[Category.DOOR])
        elif o.kind == "window":
            for p, q in window_symbol(wall, o):
                msp.add_line(p, q, dxfattribs=L[Category.WINDOW])

    for room in plan.rooms:
        poly = Polygon(room.polygon)
        msp.add_lwpolyline(room.polygon, close=True, dxfattribs=L[Category.ROOM])
        c = poly.representative_point()
        msp.add_text(room.name, height=250, dxfattribs=L[Category.ROOM]).set_placement(
            (c.x, c.y + 150), align=TextEntityAlignment.MIDDLE_CENTER)
        area = f"{poly.area / 1e6:.1f} m²".replace(".", ",")
        msp.add_text(area, height=180, dxfattribs=L[Category.ROOM]).set_placement(
            (c.x, c.y - 200), align=TextEntityAlignment.MIDDLE_CENTER)

    for t in plan.texts:
        msp.add_text(t.text, height=t.height, rotation=t.angle, dxfattribs=L[Category.TEXT]).set_placement(
            t.pos, align=TextEntityAlignment.MIDDLE_CENTER)

    for pl in plan.lines:
        if len(pl) >= 2:
            msp.add_lwpolyline(pl, dxfattribs=L[Category.OTHER])

    for chain in plan.dims:
        for p, q in zip(chain.points, chain.points[1:]):
            base = (p[0] + chain.offset[0], p[1] + chain.offset[1])
            angle = 0 if abs(q[1] - p[1]) < 1e-6 else 90
            msp.add_linear_dim(base=base, p1=p, p2=q, angle=angle, dimstyle="EZDXF", override=DIM_OVERRIDE,
                               dxfattribs=L[Category.DIMENSION]).render()
    doc.saveas(out)
