"""The full conversion: drawing image/PDF -> reconstructed plan in mm -> layered DXF.

    python -m draw2cad.convert drawing.pdf out.dxf [--no-ai] [--scale 100 --dpi 300]
"""

from __future__ import annotations

import argparse
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from draw2cad import ai
from draw2cad.dxf import iter_polygons, plan_to_dxf
from draw2cad.ocr import parse_mm, read_text
from draw2cad.openings import WallPx, find_openings
from draw2cad.plan import DimChain, Opening, Plan, Room, Text, Wall
from draw2cad.raster import Page, Word, load, pen_width, render_vectors
from draw2cad.scale import PAPER_MM, Scale, initial_scale, prior_scale, refine_with_outline
from draw2cad.walls import WallResult, extract_walls, remove_text

STANDARDS = yaml.safe_load((Path(__file__).parent / "standards/se.yaml").read_text(encoding="utf-8"))
MAX_WALL_MM = 520

ROOM_WORDS = [
    "SOVRUM", "SOV", "ALLRUM", "VARDAGSRUM", "KÖK", "MATRUM", "MATPLATS", "HALL", "ENTRÉ", "GROVENTRÉ",
    "WC", "WC/D", "WC/BAD", "BAD", "DUSCH", "TVÄTT", "KLK", "KLÄDVÅRD", "FÖRRÅD", "GARAGE", "KONTOR",
    "TEKNIK", "PANNRUM", "VIND", "ARBETSRUM", "BASTU", "UTERUM", "VERANDA", "GÄSTRUM", "KORRIDOR",
    "GROVKÖK", "PENTRY", "LEKRUM", "SKAFFERI", "STÄD", "CARPORT", "WALK", "IN", "CLOSET", "RUM", "TV-RUM",
    "FARSTU", "KLÄDKAMMARE", "TRAPPHALL", "BIBLIOTEK", "STUDIO", "ATELJÉ", "VINDSFÖRRÅD", "UTEPLATS",
]


def _plain(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.upper()) if unicodedata.category(c) != "Mn")


_ROOM_LOOKUP = {_plain(w): w for w in ROOM_WORDS}


def room_spelling(text: str) -> str | None:
    """OCR often drops Å/Ä/Ö ("TVATT"); restore the Swedish spelling for known room words."""
    parts = [p for p in text.replace("/", " / ").split() if p]
    fixed, known = [], 0
    for p in parts:
        if p == "/":
            fixed.append("/")
            continue
        base = _plain(p).rstrip(".:")
        word = _ROOM_LOOKUP.get(base)
        if word is None and base.rstrip("0123456789") in _ROOM_LOOKUP and base[-1:].isdigit():
            word = _ROOM_LOOKUP[base.rstrip("0123456789")] + " " + base[len(base.rstrip("0123456789")):]
        known += word is not None
        fixed.append(word or p.upper())
    if not known:
        return None
    return " ".join(fixed).replace(" / ", "/")


@dataclass
class Params:
    """Tunable knobs of the pipeline. Defaults are what the automatic conversion uses; the agent
    (draw2cad/agent.py) changes them when a drawing needs it."""

    region: tuple[float, float, float, float] | None = None  # page px x0, y0, x1, y1: only this part has walls
    max_wall_mm: float = 520  # thickest wall to expect
    min_wall_mm: float = 0  # drop wall pieces thinner than this (e.g. 60 to ignore furniture lines)
    bridge_mm: float = 3200  # widest door/window gap bridged when finding the building outline
    max_door_mm: float = 1300  # widest door at a wall end
    wall_source: str = "auto"  # "auto" | "raster" | "vector" (vector PDFs only)
    wall_pens: list[float] | None = None  # vector PDFs: pen widths (pt) that draw walls; None = detect
    cues: tuple[str, ...] = ("solid", "tone", "pairs", "narrow")  # which wall cues to use
    erase_dimension_lines: bool = True
    snap_to_standards: bool = True


@dataclass
class Options:
    use_ai: bool = True
    page: int = 0
    user_scale: dict | None = None  # {"mm_per_px": s} or {"scale": 100, "dpi": 300}
    params: Params = field(default_factory=Params)


@dataclass
class Result:
    plan: Plan
    page: Page
    scale: Scale
    words: list[Word]
    walls: WallResult
    walls_px: list[WallPx]
    ai: dict | None = None
    ai_error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    wall_pens: set[float] = field(default_factory=set)  # vector PDFs: pen widths the walls use
    params: Params = field(default_factory=Params)

    def to_mm(self, x: float, y: float) -> tuple[float, float]:
        H = self.page.gray.shape[0]
        return x * self.scale.sx, (H - y) * self.scale.sy

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        H = self.page.gray.shape[0]
        return x / self.scale.sx, H - y / self.scale.sy


# --- snapping to Swedish standards ---------------------------------------------------------------


def snap(value: float, choices: list[float], rel_tol: float) -> float:
    best = min(choices, key=lambda c: abs(c - value))
    return float(best) if abs(best - value) <= rel_tol * best else round(value / 5) * 5


def snap_thickness(t: float, exterior: bool) -> float:
    if exterior:
        return snap(t, STANDARDS["walls"]["exterior"], 0.2)
    return snap(t, STANDARDS["walls"]["interior_nonbearing"] + STANDARDS["walls"]["interior_bearing"], 0.3)


def snap_width(w: float, kind: str) -> float:
    if kind == "door":
        mods = STANDARDS["doors"]["interior_modules"] + STANDARDS["doors"]["exterior_modules"]
        choices = [m * 100 + 10 for m in sorted(set(mods))]
        best = min(choices, key=lambda c: abs(c - w))
        return float(best) if abs(best - w) <= 80 else round(w / 10) * 10
    if kind == "window":
        choices = [m * 100 for m in STANDARDS["windows"]["width_modules"]]
        best = min(choices, key=lambda c: abs(c - w))
        return float(best) if abs(best - w) <= 80 else round(w / 10) * 10
    return round(w / 10) * 10


def r5(v: float) -> float:
    return round(v / 5) * 5


# --- px -> mm plan ---------------------------------------------------------------------------------


@dataclass
class _W:
    horizontal: bool
    lo: float
    hi: float
    f0: float
    f1: float
    exterior: bool
    openings: list[tuple[str, float, float, int, bool]]  # kind, c0, c1, side (+1 = towards f1), hinge_lo

    @property
    def c(self) -> float:
        return (self.f0 + self.f1) / 2

    @property
    def t(self) -> float:
        return self.f1 - self.f0


def walls_to_mm(walls_px: list[WallPx], s: Scale, H: int, snap_standards: bool = True) -> list[_W]:
    out = []
    for wp in walls_px:
        b = wp.band
        if b.horizontal:  # y flips: f0/f1 swap, swing side flips, along axis (x) unchanged
            lo, hi = b.lo * s.sx, b.hi * s.sx
            f0, f1 = (H - b.f1) * s.sy, (H - b.f0) * s.sy
            outer = -b.outer
            ops = [(o.kind, o.c0 * s.sx, o.c1 * s.sx, -o.side, o.hinge_lo) for o in wp.openings]
        else:  # along axis (y) flips: lo/hi swap and the hinge end flips
            lo, hi = (H - b.hi) * s.sy, (H - b.lo) * s.sy
            f0, f1 = b.f0 * s.sx, b.f1 * s.sx
            outer = b.outer
            ops = [(o.kind, (H - o.c1) * s.sy, (H - o.c0) * s.sy, o.side, not o.hinge_lo) for o in wp.openings]
        t = snap_thickness(f1 - f0, b.exterior) if snap_standards else r5(f1 - f0)
        if b.exterior and outer == -1:
            f0 = r5(f0)
            f1 = f0 + t
        elif b.exterior and outer == 1:
            f1 = r5(f1)
            f0 = f1 - t
        else:
            mid = r5((f0 + f1) / 2)
            f0, f1 = mid - t / 2, mid + t / 2
        out.append(_W(b.horizontal, r5(lo), r5(hi), f0, f1, b.exterior, ops))
    return out


def _end_at(w: _W, end: float, others: list[_W]) -> float:
    """Where the centreline should end: on a crossing wall's centreline if this end meets one."""
    best, best_d = None, None
    for o in others:
        # At an L-corner each wall may stop at the other's inner face, so the crossing wall's
        # span only has to come within a wall thickness of this wall's centreline.
        reach = w.t + 60
        if o.horizontal == w.horizontal or not (o.lo - reach <= w.c <= o.hi + reach):
            continue
        d = abs(end - o.c)
        if d <= o.t / 2 + w.t / 2 + 60 and (best_d is None or d < best_d):
            best, best_d = o.c, d
    return best if best is not None else None


def build_plan(ws: list[_W]) -> Plan:
    plan = Plan()
    for i, w in enumerate(ws):
        a_end = _end_at(w, w.lo, ws)
        b_end = _end_at(w, w.hi, ws)
        a_al = a_end if a_end is not None else w.lo + w.t / 2
        b_al = b_end if b_end is not None else w.hi - w.t / 2
        if b_al - a_al < 1:
            continue
        a, b = ((a_al, w.c), (b_al, w.c)) if w.horizontal else ((w.c, a_al), (w.c, b_al))
        wid = f"{'E' if w.exterior else 'I'}{i}"
        plan.walls.append(Wall(wid, a, b, w.t, w.exterior))
        for kind, c0, c1, side, hinge_lo in w.openings:
            width = snap_width(c1 - c0, kind)
            center = r5((c0 + c1) / 2)
            # Left of a->b is +y for horizontal walls and -x for vertical ones.
            swing = side if w.horizontal else -side
            plan.openings.append(Opening(f"O{len(plan.openings)}", kind, wid, center - a_al, width,
                                         swing=swing or 1, hinge_at_start=hinge_lo))
    return plan


# --- rooms, text, other linework -----------------------------------------------------------------


def footprint_mm(res: Result):
    contours, _ = cv2.findContours(res.walls.footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append(Polygon([res.to_mm(float(x), float(y)) for x, y in c]).buffer(0))
    return unary_union(polys) if polys else Polygon()


def add_rooms(plan: Plan, res: Result) -> None:
    if not plan.walls:
        return
    union = unary_union([w.polygon() for w in plan.walls])
    # Spaces enclosed by walls; where an exterior wall was not recognised the building outline
    # closes the room instead, so one missing wall does not make every room leak outside.
    filled = unary_union([Polygon(p.exterior) for p in iter_polygons(union)])  # walls plus what they enclose
    region = filled.union(footprint_mm(res))
    holes = []
    for piece in iter_polygons(region.difference(union)):
        piece = piece.buffer(-100, join_style="mitre").buffer(100, join_style="mitre")  # drop slivers
        for p in iter_polygons(piece):
            if p.area < 1.0e6:
                continue
            walled = p.exterior.difference(union.buffer(60)).length
            if walled <= 0.4 * p.exterior.length:  # mostly bounded by real walls: a room
                holes.append(Polygon(p.exterior))
    used: set[int] = set()
    ai_rooms = (res.ai or {}).get("rooms", []) if res.ai else []
    H, W = res.page.gray.shape
    for h in sorted(holes, key=lambda p: -p.area):
        names = []
        for i, w in enumerate(res.words):
            name = room_spelling(w.text)
            if name and h.contains(Point(*res.to_mm(*w.center))):
                names.append((w.center[1], w.center[0], name))
                used.add(i)
        name = " ".join(n for _, _, n in sorted(names)) if names else ""
        if not name:
            for r in ai_rooms:
                if h.contains(Point(*res.to_mm(r["x"] * W, r["y"] * H))):
                    name = r["name"].upper()
                    break
        coords = [(r5(x), r5(y)) for x, y in h.exterior.coords[:-1]]
        plan.rooms.append(Room(name or "RUM", coords))
    res.plan.meta["room_words"] = sorted(used)


def add_texts(plan: Plan, res: Result) -> None:
    used = set(plan.meta.get("room_words", []))
    for i, w in enumerate(res.words):
        if i in used or not w.text.strip():
            continue
        if parse_mm(w.text) is not None:
            continue  # dimensions are regenerated from the geometry
        x, y = res.to_mm(*w.center)
        plan.texts.append(Text(w.text, (r5(x), r5(y)), max(100.0, r5(0.7 * w.size * res.scale.mean)),
                               90.0 if w.vertical else 0.0))


def add_other_lines(plan: Plan, res: Result) -> None:
    """Furniture, fixtures and stairs inside the building, as plain line segments."""
    wr = res.walls
    if res.wall_pens:  # vector PDF: take the other pens' lines exactly instead of tracing pixels
        fp = wr.footprint
        H, W = fp.shape
        boxes = [w.box for w in res.words]
        for width, pts in res.page.vectors:
            if width in res.wall_pens or len(pts) < 2:
                continue
            mx, my = pts.mean(axis=0)
            if not (0 <= mx < W and 0 <= my < H) or fp[int(my), int(mx)] == 0:
                continue
            if any(b[0] <= mx <= b[2] and b[1] <= my <= b[3] for b in boxes):
                continue  # stroked lettering
            plan.lines.append([tuple(round(v) for v in res.to_mm(x, y)) for x, y in pts])
        return
    pen = wr.pen
    keep = wr.clean.copy()
    keep[cv2.dilate(wr.mask, np.ones((3, 3), np.uint8), iterations=int(2 * pen)) > 0] = 0
    keep[wr.footprint == 0] = 0
    for w in res.words:
        x0, y0, x1, y1 = (int(v) for v in w.box)
        keep[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 0
    for wp in res.walls_px:  # door and window symbols are drawn by the plan itself
        b = wp.band
        for o in wp.openings:
            w = o.c1 - o.c0
            if b.horizontal:
                keep[max(0, int(b.f0 - w - pen)):int(b.f1 + w + pen), max(0, int(o.c0 - pen)):int(o.c1 + pen)] = 0
            else:
                keep[max(0, int(o.c0 - pen)):int(o.c1 + pen), max(0, int(b.f0 - w - pen)):int(b.f1 + w + pen)] = 0
    segs = cv2.HoughLinesP(keep, 1, np.pi / 180, threshold=int(8 * pen), minLineLength=int(6 * pen),
                           maxLineGap=int(2 * pen))
    if segs is None:
        return
    for x0, y0, x1, y1 in segs.reshape(-1, 4)[:4000]:
        a, b = res.to_mm(x0, y0), res.to_mm(x1, y1)
        plan.lines.append([(round(a[0]), round(a[1])), (round(b[0]), round(b[1]))])


def add_dimensions(plan: Plan) -> None:
    ext = [w for w in plan.walls if w.exterior]
    if not ext:
        return
    for poly in iter_polygons(unary_union([w.polygon() for w in ext]).buffer(1)):
        x0, y0, x1, y1 = (r5(v) for v in poly.bounds)
        if (x1 - x0) < 2000 or (y1 - y0) < 2000:
            continue
        plan.dims.append(DimChain([(x0, y1), (x1, y1)], (0.0, 1000.0)))
        plan.dims.append(DimChain([(x0, y0), (x0, y1)], (-1000.0, 0.0)))


# --- pipeline --------------------------------------------------------------------------------------


def _ai_scale(ai_data: dict, page: Page, outlines: list) -> Scale | None:
    note = (ai_data.get("scale_note") or "").replace(" ", "")
    if note.startswith("1:") and note[2:].isdigit():
        H, W = page.gray.shape
        paper = (ai_data.get("paper_size") or "").upper()
        dpi = page.dpi
        if dpi is None and paper in PAPER_MM:
            dpi = max(H, W) / (PAPER_MM[paper][0] / 25.4)
        if dpi:
            s = 25.4 / dpi * int(note[2:])
            return Scale(s, s, "AI: scale note", 0.55, [f"Claude read scale {note} on {paper or 'known dpi'}"])
    plans = [p for p in ai_data.get("plans", []) if p.get("overall_width_mm", 0) > 1000]
    if plans and outlines:
        x0, y0, x1, y1 = max(outlines, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        p = max(plans, key=lambda p: p["overall_width_mm"])
        sx = p["overall_width_mm"] / (x1 - x0)
        sy = p["overall_depth_mm"] / (y1 - y0) if p.get("overall_depth_mm", 0) > 1000 else sx
        return Scale(sx, sy, "AI: overall dimensions", 0.6, [f"Claude read overall {p['overall_width_mm']:.0f} mm"])
    return None


def _erase_dim_lines(img: np.ndarray, scale: Scale | None, pen: float) -> np.ndarray:
    """Dimension lines are not walls; erase them (their thin extension lines drop out by themselves)."""
    out = img.copy()
    r = int(2 * pen + 2)
    for horizontal, pos, a, b in (scale.dim_lines if scale else []):
        if horizontal:
            out[max(0, int(pos) - r):int(pos) + r + 1, max(0, int(a) - r):int(b) + r] = 0
        else:
            out[max(0, int(a) - r):int(b) + r, max(0, int(pos) - r):int(pos) + r + 1] = 0
    return out


def wall_pens(vectors, bands, shape) -> set[float]:
    """Pen widths whose lines lie on the wall faces found in the raster pass.

    The raster pass is noisy but mostly right about where walls are; CAD drawings put walls on
    their own pen, so the pens that cover those faces best are the wall pens.
    """
    faces = np.zeros(shape, np.uint8)
    for b in bands:
        x0, y0, x1, y1 = (int(v) for v in b.rect())
        cv2.rectangle(faces, (x0, y0), (x1, y1), 255, 5)
    n_faces = np.count_nonzero(faces)
    if not n_faces:
        return set()
    stats = []
    for w in sorted({w for w, _ in vectors}):
        img = render_vectors(vectors, {w}, shape)
        on = np.count_nonzero(img & faces)
        stats.append((on / n_faces, on / max(1, np.count_nonzero(img)), w))
    chosen: set[float] = set()
    covered = np.zeros(shape, np.uint8)
    for cov, prec, w in sorted(stats, reverse=True):
        if prec < 0.15:
            continue
        img = render_vectors(vectors, {w}, shape) & faces
        gain = np.count_nonzero(img & ~covered) / n_faces
        if gain >= (0.25 if not chosen else 0.15):
            chosen.add(w)
            covered |= img
    return chosen


def outlines_px(fp: np.ndarray) -> list[tuple[float, float, float, float]]:
    contours, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return []
    biggest = max(w * h for _, _, w, h in boxes)
    return [(x, y, x + w, y + h) for x, y, w, h in boxes if w * h >= 0.05 * biggest]


def merge_words(pdf_words: list[Word], ocr_words: list[Word]) -> list[Word]:
    """PDF text is exact, but CAD exports usually draw most lettering as strokes (SHX fonts), which
    only OCR can read. Keep all PDF words and every OCR word that does not repeat one of them."""
    out = list(pdf_words)
    for w in ocr_words:
        cx, cy = w.center
        if not any(p.box[0] - 2 <= cx <= p.box[2] + 2 and p.box[1] - 2 <= cy <= p.box[3] + 2 for p in pdf_words):
            out.append(w)
    return out


@dataclass
class Prepared:
    """Everything that does not depend on the scale; computed once per upload."""

    page: Page
    words: list[Word]
    ai: dict | None
    ai_error: str | None
    timings: dict[str, float]


def prepare(data: bytes, filename: str, opts: Options) -> Prepared:
    t = {}
    t0 = time.perf_counter()
    page = load(data, filename, opts.page)
    t["load"] = time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(ai.analyze_drawing, page.gray) if opts.use_ai and ai.available() else None
        t0 = time.perf_counter()
        words = merge_words(page.words, read_text(page.gray))
        t["ocr"] = time.perf_counter() - t0
        ai_data, ai_error = None, None
        if fut is not None:
            t0 = time.perf_counter()
            try:
                ai_data = fut.result()
            except Exception as e:  # the conversion must work without the AI step
                body = getattr(e, "body", None)
                err = body.get("error") if isinstance(body, dict) else None
                ai_error = err.get("message") if isinstance(err, dict) else str(e)
            t["ai"] = time.perf_counter() - t0
    if opts.use_ai and not ai.available():
        ai_error = "no ANTHROPIC_API_KEY configured"
    return Prepared(page, words, ai_data, ai_error, t)


def analyze(data: bytes, filename: str, opts: Options | None = None) -> Result:
    opts = opts or Options()
    return analyze_prepared(prepare(data, filename, opts), opts)


def _in_region(region, x: float, y: float) -> bool:
    return region is None or (region[0] <= x <= region[2] and region[1] <= y <= region[3])


def _region_only(img: np.ndarray, region, fill: int = 0) -> np.ndarray:
    if region is None:
        return img
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    out = np.full_like(img, fill)
    out[max(0, y0):y1, max(0, x0):x1] = img[max(0, y0):y1, max(0, x0):x1]
    return out


def analyze_prepared(prep: Prepared, opts: Options) -> Result:
    page, words, ai_data, ai_error = prep.page, prep.words, prep.ai, prep.ai_error
    p = opts.params
    t = dict(prep.timings)
    t0 = time.perf_counter()
    pen = pen_width(page.ink)
    clean = remove_text(page.ink, words, pen)
    scale0 = initial_scale(page, words, clean, opts.user_scale)  # scale notes may sit outside the region
    max_wall_px = p.max_wall_mm / scale0.mean if scale0 else None
    door_px = p.bridge_mm / scale0.mean if scale0 else None
    max_door_px = p.max_door_mm / scale0.mean if scale0 else None
    H, W = page.gray.shape
    room_points = [w.center for w in words if room_spelling(w.text)]
    room_points += [(r["x"] * W, r["y"] * H) for r in (ai_data or {}).get("rooms", [])]
    room_points = [q for q in room_points if _in_region(p.region, *q)]

    def erase(img, pw):
        return _erase_dim_lines(img, scale0, pw) if p.erase_dimension_lines else img

    def keep_walls(wr):
        if scale0 and p.min_wall_mm:
            wr.bands = [b for b in wr.bands if b.t * scale0.mean >= p.min_wall_mm]
        return wr

    rink, rgray, rclean = _region_only(page.ink, p.region), _region_only(page.gray, p.region, 255), _region_only(clean, p.region)
    wr = keep_walls(extract_walls(rink, rgray, words, pen, max_wall_px, erase(rclean, pen), door_px, room_points, p.cues))
    walls_px = find_openings(wr.bands, rclean, wr.mask, pen, wr.t_max, max_door_px=max_door_px)

    vector_note = None
    vectors = [(w, pts) for w, pts in page.vectors if _in_region(p.region, *pts.mean(axis=0))]
    pens: set[float] = set()
    if vectors and p.wall_source != "raster":
        pens = set(p.wall_pens) if p.wall_pens else (wall_pens(vectors, wr.bands, page.gray.shape) if wr.bands else set())
    if pens:
        # Vector PDF: redo walls and openings on only the pen(s) the walls are drawn with, which
        # leaves out symbols, cabling, furniture and hatching exactly.
        vink = render_vectors(vectors, pens, page.gray.shape)
        vclean = remove_text(vink, words, 2.0)
        wr2 = keep_walls(extract_walls(vink, 255 - vclean, words, 2.0, max_wall_px, erase(vclean, 2.0), door_px,
                                       room_points, p.cues))
        if wr2.bands or p.wall_source == "vector":
            wr = wr2
            vall = remove_text(render_vectors(vectors, {w for w, _ in vectors}, page.gray.shape), words, 2.0)
            walls_px = find_openings(wr.bands, vclean, wr.mask, 2.0, wr.t_max, window_img=vall, max_door_px=max_door_px)
            vector_note = f"vector PDF: walls read from the {', '.join(f'{q:g} pt' for q in sorted(pens))} pen lines"
    t["walls"] = time.perf_counter() - t0

    outlines = outlines_px(wr.footprint)
    warnings = []
    if scale0 is not None:
        scale = scale0 if scale0.source == "user" else refine_with_outline(scale0, words, outlines)
    else:
        scale = (_ai_scale(ai_data, page, outlines) if ai_data else None)
        if scale is None:
            ext_t = [b.t for b in wr.bands if b.exterior]
            door_px = [o.c1 - o.c0 for wp in walls_px for o in wp.openings if o.kind == "door"]
            scale = prior_scale(float(np.median(ext_t)) if ext_t else None, door_px, [])
        if scale is None:
            scale = Scale(10.0, 10.0, "unknown", 0.0, ["no scale clue found; assumed 10 mm/px"])
        warnings.append("No written dimensions or scale note were found: the scale is an estimate. "
                        "Use calibration (two points + a known distance) for an exact result.")

    res = Result(Plan(), page, scale, words, wr, walls_px, ai_data, ai_error, t, warnings,
                 pens if vector_note else set(), p)
    ws = walls_to_mm(walls_px, scale, page.gray.shape[0], p.snap_to_standards)
    plan = build_plan(ws)
    res.plan = plan
    add_rooms(plan, res)
    add_texts(plan, res)
    add_other_lines(plan, res)
    add_dimensions(plan)
    if vector_note:
        wr.style = vector_note
    plan.meta.update({"scale": {"sx": scale.sx, "sy": scale.sy, "source": scale.source,
                                "confidence": scale.confidence, "notes": scale.notes},
                      "style": wr.style, "skew_deg": page.skew_deg})
    if not plan.walls:
        warnings.append("No walls were recognised.")
    return res


def convert(image: Path, out_dxf: Path, use_ai: bool = False) -> None:
    """Benchmark entry point (offline by default so runs are reproducible and free)."""
    res = analyze(Path(image).read_bytes(), Path(image).name, Options(use_ai=use_ai))
    plan_to_dxf(res.plan, out_dxf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--scale", type=float)
    ap.add_argument("--dpi", type=float)
    args = ap.parse_args()
    user = {"scale": args.scale, "dpi": args.dpi} if args.scale and args.dpi else None
    res = analyze(args.input.read_bytes(), args.input.name, Options(use_ai=not args.no_ai, user_scale=user))
    plan_to_dxf(res.plan, args.output)
    print(f"{len(res.plan.walls)} walls, {len(res.plan.openings)} openings, {len(res.plan.rooms)} rooms; "
          f"scale {res.scale.sx:.3f}/{res.scale.sy:.3f} mm/px from {res.scale.source}")
    for n in res.scale.notes + res.warnings:
        print(" -", n)
    if res.ai_error:
        print(" - AI step failed:", res.ai_error)


if __name__ == "__main__":
    main()
