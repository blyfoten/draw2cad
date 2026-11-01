"""An Opus agent that runs the conversion pipeline as tools, looks at the result over the drawing,
and tunes or corrects it until it is good enough.

    python -m draw2cad.agent drawing.pdf out.dxf [--max-turns 25]

The agent never sees anything but what the tools return: views of the drawing (with a pixel grid
so it can point at things), the reconstruction drawn over it, the recognised text, the vector
pens of a PDF, and automatic checks (room areas against the areas written in the rooms, merged
rooms, rooms without doors, outline against written dimensions).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from draw2cad import ai
from draw2cad.convert import (Options, Params, Prepared, Result, add_dimensions, add_rooms, analyze_prepared,
                              prepare, r5)
from draw2cad.dxf import door_symbol, iter_polygons, plan_to_dxf, window_symbol
from draw2cad.ocr import parse_mm
from draw2cad.plan import Opening, Wall
from draw2cad.raster import render_vectors

MODEL = ai.MODEL  # claude-opus-5-5
VIEW_MAX = 1400  # px, long side of images sent to the model
MAX_TURNS = 25

# --- rendering views ------------------------------------------------------------------------------

COLORS = {"wall_ext": (200, 110, 0), "wall_int": (40, 150, 40), "door": (0, 110, 240), "window": (200, 0, 200),
          "room": (190, 60, 20)}


def _grid(img: np.ndarray, x0: float, y0: float, k: float) -> None:
    """Pixel-coordinate grid with labels, so the model can name positions in page pixels."""
    h, w = img.shape[:2]
    span = max(w, h) / k
    step = next(s for s in (25, 50, 100, 200, 250, 500, 1000, 2000) if span / s <= 12)
    for v in range(int(math.ceil(x0 / step) * step), int(x0 + w / k) + 1, step):
        x = int((v - x0) * k)
        cv2.line(img, (x, 0), (x, h), (210, 210, 210), 1)
        cv2.putText(img, str(v), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    for v in range(int(math.ceil(y0 / step) * step), int(y0 + h / k) + 1, step):
        y = int((v - y0) * k)
        cv2.line(img, (0, y), (w, y), (210, 210, 210), 1)
        cv2.putText(img, str(v), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)


def _crop_box(shape, region) -> tuple[int, int, int, int]:
    H, W = shape
    if not region:
        return 0, 0, W, H
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    x0, x1 = sorted((max(0, min(W, x0)), max(0, min(W, x1))))
    y0, y1 = sorted((max(0, min(H, y0)), max(0, min(H, y1))))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return 0, 0, W, H
    return x0, y0, x1, y1


def render_view(res: Result | None, gray: np.ndarray, region=None, overlay: bool = True,
                layers: set[str] | None = None, base: np.ndarray | None = None) -> tuple[bytes, str]:
    x0, y0, x1, y1 = _crop_box(gray.shape, region)
    src = gray if base is None else base
    crop = src[y0:y1, x0:x1]
    k = min(1.0, VIEW_MAX / max(crop.shape))
    img = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else crop.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if overlay and res is not None:
        img = (img.astype(np.float32) * 0.55 + 115).astype(np.uint8)  # fade the drawing under the overlay
        draw_plan(img, res, x0, y0, k, layers)
    _grid(img, x0, y0, k)
    ok, png = cv2.imencode(".png", img)
    caption = f"view of page px x {x0}..{x1}, y {y0}..{y1} (scale {k:.3f} image px per page px)"
    return png.tobytes(), caption


def draw_plan(img: np.ndarray, res: Result, x0: float, y0: float, k: float, layers: set[str] | None = None) -> None:
    plan = res.plan
    on = layers or {"wall_ext", "wall_int", "door", "window", "room"}

    def P(p):
        x, y = res.to_px(*p)
        return int(round((x - x0) * k)), int(round((y - y0) * k))

    cuts = unary_union([o.cut(plan.wall(o.wall)) for o in plan.openings]) if plan.openings else None
    for ext, key in ((True, "wall_ext"), (False, "wall_int")):
        if key not in on:
            continue
        solid = unary_union([w.polygon() for w in plan.walls if w.exterior == ext])
        if cuts is not None:
            solid = solid.difference(cuts)
        layer = img.copy()
        for poly in iter_polygons(solid):
            cv2.fillPoly(layer, [np.array([P(c) for c in poly.exterior.coords])], COLORS[key])
            for h in poly.interiors:
                cv2.fillPoly(layer, [np.array([P(c) for c in h.coords])], (255, 255, 255))
        cv2.addWeighted(layer, 0.6, img, 0.4, 0, img)
    for o in plan.openings:
        w = plan.wall(o.wall)
        if o.kind == "door" and "door" in on:
            h, l, c = door_symbol(w, o)
            cv2.line(img, P(h), P(l), COLORS["door"], 2)
            a0 = math.degrees(math.atan2(P(l)[1] - P(h)[1], P(l)[0] - P(h)[0]))
            a1 = math.degrees(math.atan2(P(c)[1] - P(h)[1], P(c)[0] - P(h)[0]))
            if (a1 - a0) % 360 > 180:  # always the quarter arc, never the long way round
                a0, a1 = a1, a0
            cv2.ellipse(img, P(h), (int(o.width / res.scale.mean * k),) * 2, 0, a0, a0 + (a1 - a0) % 360,
                        COLORS["door"], 1)
        elif o.kind == "window" and "window" in on:
            for p, q in window_symbol(w, o):
                cv2.line(img, P(p), P(q), COLORS["window"], 2)
    if "room" in on:
        for i, r in enumerate(plan.rooms):
            pts = np.array([P(c) for c in r.polygon])
            cv2.polylines(img, [pts], True, COLORS["room"], 1)
            c = Polygon(r.polygon).representative_point()
            cx, cy = P((c.x, c.y))
            label = f"R{i} {_ascii(r.name)} {Polygon(r.polygon).area / 1e6:.1f}"
            cv2.putText(img, label, (cx - 4 * len(label), cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(img, label, (cx - 4 * len(label), cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["room"], 1)


def _ascii(s: str) -> str:
    return s.translate(str.maketrans("ÅÄÖåäöÉé", "AAOaaoEe"))


# --- checks ---------------------------------------------------------------------------------------

_AREA = re.compile(r"^\d{1,3}[.,]\d{1,2}$")


def checks(res: Result) -> dict:
    """Automatic sanity checks the agent uses to judge a result (no ground truth needed)."""
    plan = res.plan
    out: dict = {"scale": {"mm_per_px": [round(res.scale.sx, 4), round(res.scale.sy, 4)],
                           "source": res.scale.source, "confidence": res.scale.confidence},
                 "walls": {"exterior": sum(w.exterior for w in plan.walls),
                           "interior": sum(not w.exterior for w in plan.walls)},
                 "openings": {k: sum(o.kind == k for o in plan.openings) for k in ("door", "window", "opening")},
                 "rooms": [], "problems": []}
    areas = []
    for w in res.words:  # areas written in the rooms, e.g. "16.0" (OCR may glue a symbol on: "16.0@")
        t = re.sub(r"(m2|m²|[^\d.,])+$", "", w.text.strip())
        if _AREA.match(t):
            areas.append((Point(*res.to_mm(*w.center)), float(t.replace(",", "."))))
    door_pts = []
    for o in plan.openings:
        if o.kind in ("door", "opening"):
            c = o.center(plan.wall(o.wall))
            door_pts.append(Point(c))
    for i, r in enumerate(plan.rooms):
        poly = Polygon(r.polygon)
        written = [a for p, a in areas if poly.contains(p)]
        names = [n for n in r.name.split(" ") if n.isalpha() and len(n) > 2]
        has_door = any(poly.exterior.distance(p) < 400 for p in door_pts)
        room = {"id": f"R{i}", "name": r.name, "area_m2": round(poly.area / 1e6, 1),
                "written_area_m2": written or None, "has_door": has_door}
        out["rooms"].append(room)
        if len(written) > 1 or len(set(names)) > 1 and len(names) >= 2:
            out["problems"].append(f"{room['id']} '{r.name}' looks like several rooms merged "
                                   f"(names {names}, written areas {written}): a wall between them is missing")
        elif written and abs(poly.area / 1e6 - written[0]) > max(0.6, 0.08 * written[0]):
            out["problems"].append(f"{room['id']} '{r.name}' is {poly.area / 1e6:.1f} m² but the drawing says "
                                   f"{written[0]} m²")
        if not has_door:
            out["problems"].append(f"{room['id']} '{r.name}' has no door or opening")
        if r.name == "RUM":
            out["problems"].append(f"{room['id']} has no name (no room label found inside it)")
    used = {i for i, w in enumerate(res.words) for r in plan.rooms
            if Polygon(r.polygon).contains(Point(*res.to_mm(*w.center)))}
    stray = [w.text for i, w in enumerate(res.words) if i not in used and _is_room_word(w.text)
             and (res.params.region is None or _inside(res.params.region, w.center))]
    if stray:
        out["problems"].append(f"room labels outside every recognised room (rooms not closed?): {stray[:10]}")
    ext = [w for w in plan.walls if w.exterior]
    if ext:
        x0, y0, x1, y1 = unary_union([w.polygon() for w in ext]).bounds
        out["outline_mm"] = [round(x1 - x0), round(y1 - y0)]
    nums = sorted({v for w in res.words if (v := parse_mm(w.text)) and v >= 1000})
    if nums:
        out["written_dimensions_mm"] = nums[-12:]
    if not plan.walls:
        out["problems"].append("no walls recognised")
    return out


def _inside(region, pt) -> bool:
    return region[0] <= pt[0] <= region[2] and region[1] <= pt[1] <= region[3]


def _is_room_word(text: str) -> bool:
    from draw2cad.convert import room_spelling

    return room_spelling(text) is not None and len(text) >= 3


# --- session and tools ----------------------------------------------------------------------------


@dataclass
class Session:
    prep: Prepared
    params: Params = field(default_factory=Params)
    user_scale: dict | None = None
    result: Result | None = None
    runs: int = 0
    edits: list[dict] = field(default_factory=list)
    done: dict | None = None

    def run(self) -> Result:
        opts = Options(use_ai=False, user_scale=self.user_scale, params=self.params)
        self.result = analyze_prepared(self.prep, opts)
        self.runs += 1
        self.edits = []
        return self.result


def _image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(png).decode("ascii")}}


def _summary(res: Result) -> str:
    c = checks(res)
    return json.dumps(c, ensure_ascii=False, separators=(",", ":"))


REGION = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
          "description": "[x0, y0, x1, y1] in page pixels (read them off the grid)"}

TOOLS = [
    {
        "name": "view_drawing",
        "description": "Look at the input drawing (or part of it), optionally with the current reconstruction "
                       "drawn over it. Exterior walls blue, interior green, doors orange, windows magenta, rooms "
                       "outlined with id, name and area in m². The grid shows page pixel coordinates. Zoom in "
                       "(small region) to check details such as wall ends, door swings and window positions.",
        "input_schema": {"type": "object", "properties": {
            "region": REGION,
            "overlay": {"type": "boolean", "description": "Draw the current reconstruction over the drawing"},
            "layers": {"type": "array", "items": {"type": "string", "enum": ["wall_ext", "wall_int", "door", "window", "room"]}},
        }, "additionalProperties": False},
    },
    {
        "name": "run_conversion",
        "description": "Run the automatic conversion with the given parameters (omitted ones keep their current "
                       "value) and get the automatic checks plus an overview with the result drawn over the "
                       "drawing. Running again discards earlier manual edits, so tune parameters first and "
                       "edit last. Parameters: region = only look for walls in this part of the sheet (use it "
                       "when the sheet has a legend, title block, several plans or detail drawings); "
                       "max_wall_mm = thickest wall to expect (lower it if furniture or counters become walls, "
                       "raise it for thick old walls); min_wall_mm = drop thinner wall pieces; bridge_mm = "
                       "widest gap closed when finding the building outline; max_door_mm = widest door at a "
                       "wall end; wall_source = auto/raster/vector (vector PDFs); wall_pens = the PDF pen widths "
                       "that draw walls (see list_pens); cues = which wall cues to use: solid (thick black "
                       "strokes), tone (grey fill), pairs (two parallel face lines), narrow (narrow enclosed "
                       "spaces between lines); scale_mm_per_px = force a scale (only if the automatic one is "
                       "wrong and you have a measured reason, e.g. a written dimension between two points).",
        "input_schema": {"type": "object", "properties": {
            "region": {"anyOf": [REGION, {"type": "null"}]},
            "max_wall_mm": {"type": "number"}, "min_wall_mm": {"type": "number"},
            "bridge_mm": {"type": "number"}, "max_door_mm": {"type": "number"},
            "wall_source": {"type": "string", "enum": ["auto", "raster", "vector"]},
            "wall_pens": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
            "cues": {"type": "array", "items": {"type": "string", "enum": ["solid", "tone", "pairs", "narrow"]}},
            "erase_dimension_lines": {"type": "boolean"},
            "snap_to_standards": {"type": "boolean"},
            "scale_mm_per_px": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        }, "additionalProperties": False},
    },
    {
        "name": "read_text",
        "description": "Text recognised on the drawing (OCR and PDF text) with page-pixel positions, optionally "
                       "only inside a region. Use it to find room names, written areas, dimensions and the "
                       "scale note.",
        "input_schema": {"type": "object", "properties": {"region": REGION}, "additionalProperties": False},
    },
    {
        "name": "list_pens",
        "description": "Vector PDFs only: the pen widths used on the sheet with their total line length, and how "
                       "much of each lies on the walls found so far. CAD prints draw walls with their own pen.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "view_pens",
        "description": "Vector PDFs only: draw only the lines of the given pen widths, to see what each pen holds.",
        "input_schema": {"type": "object", "properties": {
            "pens": {"type": "array", "items": {"type": "number"}}, "region": REGION},
            "required": ["pens"], "additionalProperties": False},
    },
    {
        "name": "edit_plan",
        "description": "Fix the current reconstruction by hand where the automatic result is wrong and no "
                       "parameter fixes it. Coordinates are page pixels; sizes are mm. Actions: "
                       "add_wall {x0,y0,x1,y1 = centreline end points, thickness_mm, exterior}; "
                       "remove_walls {region}: removes walls whose centreline midpoint is inside; "
                       "add_opening {kind door|window|opening, x, y = centre on a wall, width_mm, "
                       "swing_x, swing_y = a point on the side the door opens towards (doors)}; "
                       "remove_openings {region}; rename_room {x, y = a point inside the room, name}. "
                       "Rooms are recomputed after wall edits. Only add what is actually drawn.",
        "input_schema": {"type": "object", "properties": {"actions": {"type": "array", "items": {
            "type": "object", "properties": {
                "op": {"type": "string", "enum": ["add_wall", "remove_walls", "add_opening", "remove_openings", "rename_room"]},
                "x0": {"type": "number"}, "y0": {"type": "number"}, "x1": {"type": "number"}, "y1": {"type": "number"},
                "x": {"type": "number"}, "y": {"type": "number"}, "thickness_mm": {"type": "number"},
                "exterior": {"type": "boolean"}, "region": REGION,
                "kind": {"type": "string", "enum": ["door", "window", "opening"]}, "width_mm": {"type": "number"},
                "swing_x": {"type": "number"}, "swing_y": {"type": "number"}, "name": {"type": "string"},
            }, "required": ["op"], "additionalProperties": False}}},
            "required": ["actions"], "additionalProperties": False},
    },
    {
        "name": "finish",
        "description": "Stop: the reconstruction is as good as you can get it. Give an honest quality rating "
                       "and a short summary for the user of what was done and what is still wrong.",
        "input_schema": {"type": "object", "properties": {
            "quality": {"type": "string", "enum": ["good", "acceptable", "poor"]},
            "summary": {"type": "string"}}, "required": ["quality", "summary"], "additionalProperties": False},
    },
]

SYSTEM = """You turn scanned or printed architectural drawings (mostly Swedish house plans) into accurate CAD.
A conversion pipeline does the measuring; you are the architect who checks its work and steers it.

Work like this:
1. Look at the whole drawing first (view_drawing without overlay). Note what kind of drawing it is, which
   part of the sheet is the plan you should convert (ignore legends, title blocks, sections, detail
   drawings; if there are several floor plans, convert the main/first one unless the sheet clearly has one),
   how walls are drawn (filled, outlined, hatched, grey), the scale note, written dimensions, room names
   and written room areas.
2. Run the conversion. Then compare: view the overlay of the whole plan, then zoom in on anything the
   checks flag and on every area you have doubts about. The checks compare room areas with the areas
   written in the rooms, find merged rooms, rooms without doors and room labels outside all rooms.
3. Fix the cause with parameters when the error is systematic (wrong region, furniture or stairs taken
   as walls, thin walls missed, wrong pen in a vector PDF), and re-run. Typical moves: set region to the
   plan; for a vector PDF look at list_pens/view_pens and set wall_pens; drop a cue that creates junk; raise
   min_wall_mm to remove furniture.
4. When the remaining errors are local (a missing wall stub, a wardrobe that should be a partition wall, a
   false window, a door on the wrong side), fix them with edit_plan, then check the overlay again.
5. Finish when walls, doors, windows and rooms match the drawing, or when further work would only be
   guessing. Never invent geometry that is not drawn; never change the scale without a measured reason.

Be efficient: a handful of conversion runs and targeted zooms is normally enough. Keep your messages short;
the user sees them as a progress log."""


def run_tool(session: Session, name: str, args: dict) -> list[dict]:
    """Execute one tool call; returns tool_result content blocks."""
    page = session.prep.page
    res = session.result
    if name == "view_drawing":
        overlay = args.get("overlay", False)
        if overlay and res is None:
            return [{"type": "text", "text": "No conversion yet: call run_conversion first."}]
        layers = set(args["layers"]) if args.get("layers") else None
        png, cap = render_view(res, page.gray, args.get("region"), overlay, layers)
        return [{"type": "text", "text": cap}, _image_block(png)]

    if name == "run_conversion":
        p = session.params
        changes = {}
        for f in fields(Params):
            if f.name in args:
                v = args[f.name]
                if f.name == "cues" and v is not None:
                    v = tuple(v)
                if f.name == "region" and v is not None:
                    v = tuple(float(x) for x in v)
                setattr(p, f.name, v)
                changes[f.name] = v
        if "scale_mm_per_px" in args:
            s = args["scale_mm_per_px"]
            session.user_scale = {"mm_per_px": float(s)} if s else None
        t0 = time.perf_counter()
        res = session.run()
        png, cap = render_view(res, page.gray, p.region or _plan_region(res), True)
        text = (f"Conversion #{session.runs} in {time.perf_counter() - t0:.1f}s with params "
                f"{json.dumps(asdict(p), default=str)}.\nWall style: {res.walls.style}. Scale notes: "
                f"{res.scale.notes}.\nChecks:\n{_summary(res)}")
        return [{"type": "text", "text": text}, {"type": "text", "text": cap}, _image_block(png)]

    if name == "read_text":
        region = args.get("region")
        rows = []
        for w in session.prep.words:
            if region and not _inside(region, w.center):
                continue
            rows.append(f"{w.text!r} at ({w.center[0]:.0f}, {w.center[1]:.0f}) box "
                        f"[{w.box[0]:.0f},{w.box[1]:.0f},{w.box[2]:.0f},{w.box[3]:.0f}]"
                        + (" vertical" if w.vertical else ""))
        return [{"type": "text", "text": f"{len(rows)} text items:\n" + "\n".join(rows[:250])}]

    if name == "list_pens":
        if not page.vectors:
            return [{"type": "text", "text": "Not a vector PDF (no vector linework): raster methods only."}]
        stats: dict[float, list] = {}
        for w, pts in page.vectors:
            L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            st = stats.setdefault(w, [0, 0.0])
            st[0] += 1
            st[1] += L
        faces = np.zeros(page.gray.shape, np.uint8)
        if res is not None:
            for wp in res.walls_px:
                x0, y0, x1, y1 = (int(v) for v in wp.band.rect())
                cv2.rectangle(faces, (x0, y0), (x1, y1), 255, 5)
        lines = []
        for w in sorted(stats):
            on = ""
            if res is not None and faces.any():
                img = render_vectors(page.vectors, {w}, page.gray.shape)
                on = f", {np.count_nonzero(img & faces) / max(1, np.count_nonzero(img)):.0%} of it on current walls"
            label = "fill outlines" if w < 0 else f"{w:g} pt"
            lines.append(f"pen {w:g} ({label}): {stats[w][0]} pieces, {stats[w][1]:.0f} px total{on}")
        chosen = sorted(res.wall_pens) if res is not None else []
        return [{"type": "text", "text": "\n".join(lines) + f"\nPens used for walls in the current run: {chosen}"}]

    if name == "view_pens":
        if not page.vectors:
            return [{"type": "text", "text": "Not a vector PDF."}]
        img = render_vectors(page.vectors, set(args["pens"]), page.gray.shape)
        png, cap = render_view(None, page.gray, args.get("region"), False, base=255 - img)
        return [{"type": "text", "text": f"pens {args['pens']} only; " + cap}, _image_block(png)]

    if name == "edit_plan":
        if res is None:
            return [{"type": "text", "text": "No conversion yet: call run_conversion first."}]
        notes = [apply_edit(res, a) for a in args["actions"]]
        session.edits += args["actions"]
        recompute_rooms(res)
        png, cap = render_view(res, page.gray, session.params.region or _plan_region(res), True)
        return [{"type": "text", "text": "\n".join(notes) + "\nChecks after edits:\n" + _summary(res)},
                {"type": "text", "text": cap}, _image_block(png)]

    if name == "finish":
        session.done = args
        return [{"type": "text", "text": "Finished."}]
    return [{"type": "text", "text": f"Unknown tool {name}"}]


def _plan_region(res: Result):
    ys, xs = np.nonzero(res.walls.footprint)
    if len(xs) == 0:
        return None
    m = 0.08 * max(np.ptp(xs), np.ptp(ys))
    return (xs.min() - m, ys.min() - m, xs.max() + m, ys.max() + m)


# --- manual edits ----------------------------------------------------------------------------------


def apply_edit(res: Result, a: dict) -> str:
    plan = res.plan
    op = a.get("op")
    try:
        if op == "add_wall":
            p0 = res.to_mm(a["x0"], a["y0"])
            p1 = res.to_mm(a["x1"], a["y1"])
            # Keep walls axis-aligned when they nearly are.
            if abs(p1[0] - p0[0]) < abs(p1[1] - p0[1]) * 0.05:
                p1 = (p0[0], p1[1])
            elif abs(p1[1] - p0[1]) < abs(p1[0] - p0[0]) * 0.05:
                p1 = (p1[0], p0[1])
            p0, p1 = (r5(p0[0]), r5(p0[1])), (r5(p1[0]), r5(p1[1]))
            wid = f"A{len(plan.walls)}"
            plan.walls.append(Wall(wid, p0, p1, float(a.get("thickness_mm", 95)), bool(a.get("exterior", False))))
            return f"added wall {wid} {p0}->{p1} mm"
        if op == "remove_walls":
            region = a["region"]
            gone = {w.id for w in plan.walls
                    if _inside(region, res.to_px((w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2))}
            plan.walls = [w for w in plan.walls if w.id not in gone]
            plan.openings = [o for o in plan.openings if o.wall not in gone]
            return f"removed {len(gone)} walls"
        if op == "add_opening":
            pt = Point(res.to_mm(a["x"], a["y"]))
            wall = min(plan.walls, key=lambda w: LineString([w.a, w.b]).distance(pt))
            line = LineString([wall.a, wall.b])
            if line.distance(pt) > wall.thickness + 300:
                return "add_opening: no wall near that point"
            offset = line.project(pt)
            swing = 1
            if "swing_x" in a and "swing_y" in a:
                sx, sy = res.to_mm(a["swing_x"], a["swing_y"])
                ux, uy = wall.direction
                swing = 1 if (-uy * (sx - wall.a[0]) + ux * (sy - wall.a[1])) >= 0 else -1
            plan.openings.append(Opening(f"E{len(plan.openings)}", a.get("kind", "door"), wall.id, r5(offset),
                                         float(a.get("width_mm", 910)), swing=swing))
            return f"added {a.get('kind', 'door')} on wall {wall.id}"
        if op == "remove_openings":
            region = a["region"]
            before = len(plan.openings)
            plan.openings = [o for o in plan.openings
                             if not _inside(region, res.to_px(*o.center(plan.wall(o.wall))))]
            return f"removed {before - len(plan.openings)} openings"
        if op == "rename_room":
            pt = Point(res.to_mm(a["x"], a["y"]))
            for r in plan.rooms:
                if Polygon(r.polygon).contains(pt):
                    r.name = a["name"].upper()
                    r.name_fixed = True  # type: ignore[attr-defined]
                    return f"renamed room to {r.name}"
            return "rename_room: no room at that point"
    except (KeyError, ValueError) as e:
        return f"{op}: missing or bad argument {e}"
    return f"unknown edit {op}"


def recompute_rooms(res: Result) -> None:
    plan = res.plan
    fixed = [(Polygon(r.polygon).representative_point(), r.name) for r in plan.rooms
             if getattr(r, "name_fixed", False)]
    plan.rooms, plan.dims = [], []
    add_rooms(plan, res)
    add_dimensions(plan)
    for pt, name in fixed:  # keep names the agent set
        for r in plan.rooms:
            if Polygon(r.polygon).contains(pt):
                r.name = name
                r.name_fixed = True  # type: ignore[attr-defined]


# --- the loop --------------------------------------------------------------------------------------


Event = Callable[[dict], None]


def run_agent(prep: Prepared, on_event: Event | None = None, max_turns: int = MAX_TURNS,
              session: Session | None = None) -> Session:
    """Let Opus drive the pipeline until it calls finish (or the turn limit is hit)."""
    import anthropic

    ai.available()  # loads the key from .env if needed
    client = anthropic.Anthropic()
    session = session or Session(prep)
    emit = on_event or (lambda e: None)
    page = prep.page
    intro = (f"Convert this drawing. Page {page.gray.shape[1]} x {page.gray.shape[0]} px"
             f"{', vector PDF' if page.vectors else ''}"
             f"{f', sheet {page.paper_mm[0]:.0f} x {page.paper_mm[1]:.0f} mm' if page.paper_mm else ''}"
             f"{f', {page.dpi:.0f} dpi' if page.dpi else ''}. {len(prep.words)} text items were recognised.")
    if prep.ai:
        intro += f" A first quick reading of the sheet: {json.dumps(prep.ai, ensure_ascii=False)}"
    messages: list[dict] = [{"role": "user", "content": intro}]
    usage = {"input": 0, "output": 0, "cache_read": 0}
    for turn in range(max_turns):
        emit({"type": "turn", "turn": turn + 1})
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            tools=TOOLS,
            cache_control={"type": "ephemeral"},
            output_config={"effort": "high"},
            messages=messages,
        )
        u = response.usage
        usage["input"] += u.input_tokens + (u.cache_creation_input_tokens or 0)
        usage["output"] += u.output_tokens
        usage["cache_read"] += u.cache_read_input_tokens or 0
        messages.append({"role": "assistant", "content": response.content})
        for b in response.content:
            if b.type == "text" and b.text.strip():
                emit({"type": "say", "text": b.text.strip()})
        if response.stop_reason == "refusal":
            emit({"type": "error", "text": "The model declined to continue."})
            break
        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:
            if response.stop_reason == "end_turn" and session.done is None:
                messages.append({"role": "user", "content": "Continue, and call finish when you are done."})
                continue
            break
        results = []
        for call in calls:
            emit({"type": "tool", "name": call.name, "input": call.input})
            try:
                content = run_tool(session, call.name, dict(call.input))
                is_error = False
            except Exception as e:  # report to the model, let it adapt
                content = [{"type": "text", "text": f"Tool failed: {type(e).__name__}: {e}"}]
                is_error = True
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error})
            if call.name == "run_conversion" and not is_error:
                emit({"type": "result", "runs": session.runs})
        messages.append({"role": "user", "content": results})
        if session.done is not None:
            break
    if session.result is None:  # the agent never ran a conversion; fall back to the default one
        session.run()
    session.usage = usage  # type: ignore[attr-defined]
    # Opus 5.5: $4 / $20 per MTok, cache reads $0.20 (cache writes counted as plain input: a slight underestimate)
    cost = usage["input"] * 4e-6 + usage["output"] * 20e-6 + usage["cache_read"] * 0.2e-6
    emit({"type": "done", "summary": (session.done or {}).get("summary", "Stopped at the turn limit."),
          "quality": (session.done or {}).get("quality", "unknown"), "usage": usage, "cost_usd": round(cost, 2)})
    return session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = ap.parse_args()
    prep = prepare(args.input.read_bytes(), args.input.name, Options(use_ai=True))

    def show(e):
        if e["type"] == "say":
            print("  agent:", e["text"])
        elif e["type"] == "tool":
            print("  tool:", e["name"], json.dumps(e["input"], ensure_ascii=False)[:200])
        elif e["type"] == "done":
            print(f"done ({e['quality']}, ~${e['cost_usd']}): {e['summary']}")

    s = run_agent(prep, show, args.max_turns)
    plan_to_dxf(s.result.plan, args.output)
    png, _ = render_view(s.result, prep.page.gray, s.params.region or _plan_region(s.result), True)
    args.output.with_suffix(".png").write_bytes(png)
    print("wrote", args.output, "and", args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
