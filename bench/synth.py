"""Synthetic Swedish-style floor plans with exact ground truth, rendered and degraded like real scans.

    python -m bench.synth --out bench/cases --count 12 --seed 1

Each case directory gets: image.png (the input), gt.json (plan in mm + render transform),
gt.dxf (ground truth as a clean drawing, for viewing).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from shapely.ops import unary_union

from draw2cad.dxf import door_symbol, iter_polygons, plan_to_dxf, window_symbol
from draw2cad.plan import DimChain, Opening, Plan, Room, Wall, rect

STANDARDS = yaml.safe_load((Path(__file__).parents[1] / "draw2cad/standards/se.yaml").read_text(encoding="utf-8"))

ROOM_NAMES_LARGE = ["ALLRUM", "VARDAGSRUM", "KÖK", "KÖK/MATRUM"]
ROOM_NAMES_MEDIUM = ["SOVRUM", "SOVRUM", "KONTOR", "SOV 2", "SOV 3", "GROVENTRÉ"]
ROOM_NAMES_SMALL = ["WC/D", "HALL", "KLK", "TVÄTT", "BAD", "FÖRRÅD", "ENTRÉ"]


def door_width(modules: int) -> float:
    return modules * 100 + 10


# --- plan generation -------------------------------------------------------------------------------


def generate_plan(rng: random.Random) -> Plan:
    W = rng.randrange(8000, 15001, 100)  # outer face to outer face
    D = rng.randrange(6000, 10001, 100)
    t_ext = rng.choice(STANDARDS["walls"]["exterior"])
    h = t_ext / 2
    x0, y0, x1, y1 = h, h, W - h, D - h  # exterior centerline rectangle

    walls = [
        Wall("E0", (x0, y0), (x1, y0), t_ext, True),  # south
        Wall("E1", (x1, y0), (x1, y1), t_ext, True),  # east
        Wall("E2", (x1, y1), (x0, y1), t_ext, True),  # north
        Wall("E3", (x0, y1), (x0, y0), t_ext, True),  # west
    ]
    interior_choices = STANDARDS["walls"]["interior_nonbearing"] + STANDARDS["walls"]["interior_bearing"][:1]
    min_room = 2400.0
    leaves: list[tuple[float, float, float, float]] = []

    def split(r: tuple[float, float, float, float], depth: int) -> None:
        rx0, ry0, rx1, ry1 = r
        w, d = rx1 - rx0, ry1 - ry0
        area = w * d / 1e6
        if depth > 0 and (area < rng.uniform(9, 22) or depth >= 4):
            leaves.append(r)
            return
        vertical = w > d if abs(w - d) > 1500 else rng.random() < 0.5
        span = w if vertical else d
        if span < 2 * min_room:
            leaves.append(r)
            return
        c = round(rng.uniform(min_room, span - min_room) / 100) * 100
        t = rng.choice(interior_choices)
        idx = f"I{len(walls)}"
        if vertical:
            walls.append(Wall(idx, (rx0 + c, ry0), (rx0 + c, ry1), t, False))
            split((rx0, ry0, rx0 + c, ry1), depth + 1)
            split((rx0 + c, ry0, rx1, ry1), depth + 1)
        else:
            walls.append(Wall(idx, (rx0, ry0 + c), (rx1, ry0 + c), t, False))
            split((rx0, ry0, rx1, ry0 + c), depth + 1)
            split((rx0, ry0 + c, rx1, ry1), depth + 1)

    split((x0, y0, x1, y1), 0)
    plan = Plan(walls=walls)
    plan.openings = place_openings(rng, plan)
    plan.rooms = make_rooms(rng, plan, leaves)
    plan.dims = make_dims(plan, W, D, t_ext)
    plan.meta = {"outer_width": W, "outer_depth": D, "t_ext": t_ext}
    return plan


def junctions_along(plan: Plan, wall: Wall) -> list[float]:
    """Offsets along `wall` where other walls meet it."""
    out = []
    ux, uy = wall.direction
    for other in plan.walls:
        if other is wall:
            continue
        for p in (other.a, other.b):
            dx, dy = p[0] - wall.a[0], p[1] - wall.a[1]
            along, across = dx * ux + dy * uy, abs(-dx * uy + dy * ux)
            if across < 1 and -1 <= along <= wall.length + 1:
                out.append(along)
        # walls crossing through this one (T-junction from the other side is covered above)
    return out


def free_offset(rng: random.Random, plan: Plan, wall: Wall, width: float, taken: list[tuple[float, float]],
                tries: int = 30) -> float | None:
    blocked = [(j - width / 2 - 250, j + width / 2 + 250) for j in junctions_along(plan, wall)]
    blocked += [(lo - width / 2 - 300, hi + width / 2 + 300) for lo, hi in taken]
    lo, hi = width / 2 + 300, wall.length - width / 2 - 300
    if hi <= lo:
        return None
    for _ in range(tries):
        c = round(rng.uniform(lo, hi) / 10) * 10
        if all(not (a < c < b) for a, b in blocked):
            return c
    return None


def place_openings(rng: random.Random, plan: Plan) -> list[Opening]:
    openings: list[Opening] = []
    taken: dict[str, list[tuple[float, float]]] = {}

    def add(kind: str, wall: Wall, width: float) -> bool:
        c = free_offset(rng, plan, wall, width, taken.get(wall.id, []))
        if c is None:
            return False
        taken.setdefault(wall.id, []).append((c - width / 2, c + width / 2))
        openings.append(Opening(f"O{len(openings)}", kind, wall.id, c, width,
                                swing=rng.choice((1, -1)), hinge_at_start=rng.random() < 0.5))
        return True

    for wall in plan.walls:
        if not wall.exterior:
            add("door", wall, door_width(rng.choice(STANDARDS["doors"]["interior_modules"][1:])))
    ext = [w for w in plan.walls if w.exterior]
    for wall in rng.sample(ext, k=rng.choice((1, 2))):
        add("door", wall, door_width(rng.choice(STANDARDS["doors"]["exterior_modules"])))
    for wall in ext:
        for _ in range(int(wall.length // 3500) + rng.randint(0, 1)):
            add("window", wall, rng.choice(STANDARDS["windows"]["width_modules"]) * 100)
    # Exterior doors swing outwards: the outside is to the right of the counter-clockwise loop.
    for o in openings:
        if o.kind == "door" and plan.wall(o.wall).exterior:
            o.swing = -1
    return openings


def make_rooms(rng: random.Random, plan: Plan, leaves) -> list[Room]:
    walls_only = unary_union([w.polygon() for w in plan.walls])
    rooms = []
    for r in sorted(leaves, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1])):
        piece = rect(*r).difference(walls_only)
        polys = sorted(iter_polygons(piece), key=lambda p: -p.area)
        if not polys:
            continue
        area = polys[0].area / 1e6
        pool = ROOM_NAMES_LARGE if area > 20 else ROOM_NAMES_MEDIUM if area > 8 else ROOM_NAMES_SMALL
        rooms.append(Room(rng.choice(pool), [tuple(map(float, c)) for c in polys[0].exterior.coords[:-1]]))
    return rooms


def make_dims(plan: Plan, W: float, D: float, t_ext: float) -> list[DimChain]:
    # Overall dimensions to outer faces, plus a chain along the south side to interior wall centres.
    south = plan.wall("E0")
    stops = sorted({round(o) for o in junctions_along(plan, south) if 1 < o < south.length - 1})
    xs = [0.0] + [south.a[0] + s for s in stops] + [W]
    return [
        DimChain([(0.0, D), (W, D)], (0.0, 1200.0)),
        DimChain([(x, 0.0) for x in xs], (0.0, -1200.0)),
        DimChain([(0.0, 0.0), (0.0, D)], (-1200.0, 0.0)),
    ]


# --- rendering -------------------------------------------------------------------------------------


@dataclass
class RenderSpec:
    scale: int  # drawing scale 1:scale
    dpi: int
    style: str  # "filled" | "outlined" | "hatched"
    margin_mm: float = 3500.0  # world mm around the building

    @property
    def px_per_mm(self) -> float:
        return self.dpi / 25.4 / self.scale


FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size_px: int):
    # Pillow's built-in font has no Å/Ä/Ö, so prefer a system font that does.
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, max(8, size_px))
    return ImageFont.load_default(size=max(8, size_px))


def fmt_mm(v: float) -> str:
    s = f"{int(round(v)):,}".replace(",", " ")
    return s


def render(plan: Plan, spec: RenderSpec) -> tuple[np.ndarray, np.ndarray]:
    """Render to a grayscale image. Returns (image, A) where A maps world mm [x, y, 1] -> pixel [u, v]."""
    bx0, by0, bx1, by1 = plan.bounds()
    m = spec.margin_mm
    ss = 2 if spec.px_per_mm * (bx1 - bx0 + 2 * m) < 4000 else 1  # supersample for anti-aliasing
    k = spec.px_per_mm * ss
    width, height = int((bx1 - bx0 + 2 * m) * k), int((by1 - by0 + 2 * m) * k)

    def P(p) -> tuple[float, float]:
        return (p[0] - bx0 + m) * k, (by1 + m - p[1]) * k

    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    pen = max(1, round(0.5 * spec.dpi / 300 * ss * 2))  # ~0.35-0.5 mm on paper
    thin = max(1, pen // 2)

    solid = plan.wall_solid()
    for poly in iter_polygons(solid):
        ext = [P(c) for c in poly.exterior.coords]
        holes = [[P(c) for c in h.coords] for h in poly.interiors]
        if spec.style == "filled":
            d.polygon(ext, fill=0)
            for h in holes:
                d.polygon(h, fill=255)
        else:
            d.line(ext, fill=0, width=pen, joint="curve")
            for h in holes:
                d.line(h, fill=0, width=pen, joint="curve")
    if spec.style == "hatched":
        mask = Image.new("L", (width, height), 0)
        md = ImageDraw.Draw(mask)
        for poly in iter_polygons(solid):
            md.polygon([P(c) for c in poly.exterior.coords], fill=255)
            for h in poly.interiors:
                md.polygon([P(c) for c in h.coords], fill=0)
        hatch = Image.new("L", (width, height), 255)
        hd = ImageDraw.Draw(hatch)
        gap = max(4, int(80 * k))
        for x in range(-height, width, gap):
            hd.line([(x, height), (x + height, 0)], fill=0, width=thin)
        img.paste(hatch, mask=mask)

    for o in plan.openings:
        wall = plan.wall(o.wall)
        if o.kind == "door":
            hinge, leaf, closed = door_symbol(wall, o)
            d.line([P(hinge), P(leaf)], fill=0, width=thin)
            hx, hy = P(hinge)
            r = o.width * k
            a0 = math.degrees(math.atan2(P(leaf)[1] - hy, P(leaf)[0] - hx))
            a1 = math.degrees(math.atan2(P(closed)[1] - hy, P(closed)[0] - hx))
            if (a1 - a0) % 360 > 180:
                a0, a1 = a1, a0
            d.arc([hx - r, hy - r, hx + r, hy + r], a0, a1, fill=0, width=thin)
        else:
            for p, q in window_symbol(wall, o):
                d.line([P(p), P(q)], fill=0, width=thin)

    text_px = int(250 * k)  # 2.5 mm text at 1:100
    font = _font(text_px)
    for room in plan.rooms:
        c = Polygon(room.polygon).representative_point()
        d.text(P((c.x, c.y)), room.name, fill=0, font=font, anchor="mm")

    tick = 120 * k
    for chain in plan.dims:
        pts = [(p[0] + chain.offset[0], p[1] + chain.offset[1]) for p in chain.points]
        d.line([P(pts[0]), P(pts[-1])], fill=0, width=thin)
        for p, q in zip(chain.points, pts):
            d.line([P(p), P(q)], fill=0, width=thin)  # extension line
            u, v = P(q)
            d.line([(u - tick, v + tick), (u + tick, v - tick)], fill=0, width=thin)  # architectural tick
        horizontal = abs(pts[0][1] - pts[-1][1]) < 1e-6
        for p, q, a, b in zip(chain.points, chain.points[1:], pts, pts[1:]):
            value = math.dist(p, q)
            mid = P(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            label = fmt_mm(value)
            if horizontal:
                d.text((mid[0], mid[1] - text_px * 0.3), label, fill=0, font=font, anchor="md")
            else:
                tw = int(d.textlength(label, font=font)) + 4
                tmp = Image.new("L", (tw, text_px * 2), 255)
                ImageDraw.Draw(tmp).text((tw / 2, text_px * 1.6), label, fill=0, font=font, anchor="md")
                tmp = tmp.rotate(90, expand=True, fillcolor=255)
                img.paste(tmp, (int(mid[0] - tmp.width), int(mid[1] - tmp.height / 2)))

    if ss > 1:
        img = img.resize((width // ss, height // ss), Image.Resampling.LANCZOS)
    kk = k / ss
    A = np.array([[kk, 0, (m - bx0) * kk], [0, -kk, (by1 + m) * kk]])
    return np.asarray(img), A


# --- degradation -----------------------------------------------------------------------------------


def degrade(img: np.ndarray, A: np.ndarray, rng: random.Random, level: str) -> tuple[np.ndarray, np.ndarray]:
    """Simulate print/scan/print-to-PDF damage. Returns the new image and the updated world->pixel map."""
    if level == "clean":
        return img, A
    h, w = img.shape
    rough = level == "rough"
    # Anisotropic scale (printer/scanner feed), small rotation, optional slight perspective.
    sx = 1 + rng.uniform(-0.012, 0.012)
    sy = 1 + rng.uniform(-0.012, 0.012)
    ang = math.radians(rng.uniform(-2.5, 2.5) if rough else rng.uniform(-1.0, 1.0))
    c, s = math.cos(ang), math.sin(ang)
    M = np.array([[sx * c, -sy * s, 0], [sx * s, sy * c, 0], [0, 0, 1]])
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    moved = M @ corners
    tx, ty = -moved[0].min() + 20, -moved[1].min() + 20
    M[0, 2], M[1, 2] = tx, ty
    out_w, out_h = int(np.ptp(moved[0]) + 40), int(np.ptp(moved[1]) + 40)
    if rough:
        jitter = 0.004 * max(w, h)
        src = np.float32([[0, 0], [out_w, 0], [0, out_h], [out_w, out_h]])
        dst = src + np.float32([[rng.uniform(-jitter, jitter) for _ in range(2)] for _ in range(4)])
        Hp = cv2.getPerspectiveTransform(src, dst)
        H = Hp @ M
        warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=255)
    else:
        H = M
        warped = cv2.warpAffine(img, M[:2], (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=255)

    out = warped.astype(np.float32)
    # Print-to-PDF / low-res quantisation: down-sample, then nearest up-sample.
    q = rng.uniform(0.45, 0.8)
    small = cv2.resize(out, (int(out_w * q), int(out_h * q)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    # Toner/blur, paper tone with gradient, noise, specks.
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.3 if rough else 0.9))
    grad = np.linspace(rng.uniform(0, 25), rng.uniform(0, 25), out_w, dtype=np.float32)[None, :]
    out = out * rng.uniform(0.85, 0.97) + 10 - grad * 0.5
    out += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6 if rough else 3, out.shape)
    if rough:
        n = int(out_w * out_h * 2e-5)
        ys = np.random.default_rng(rng.randrange(1 << 30)).integers(0, out_h, n)
        xs = np.random.default_rng(rng.randrange(1 << 30)).integers(0, out_w, n)
        out[ys, xs] = 30
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 80)])
    out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    A3 = np.vstack([A, [0, 0, 1]])
    return out, H @ A3  # 3x3 projective world->pixel


# --- cases -----------------------------------------------------------------------------------------


def make_case(out_dir: Path, seed: int, level: str) -> None:
    rng = random.Random(seed)
    plan = generate_plan(rng)
    spec = RenderSpec(scale=rng.choice((50, 100)), dpi=rng.choice((150, 200, 300)),
                      style=rng.choice(("filled", "outlined", "hatched")))
    img, A = render(plan, spec)
    img, T = degrade(img, A, rng, level)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "image.png"), img)
    plan.meta.update({"seed": seed, "level": level, "scale": spec.scale, "dpi": spec.dpi, "style": spec.style,
                      "world_to_px": np.asarray(T).tolist()})
    plan.to_json(out_dir / "gt.json")
    plan_to_dxf(plan, out_dir / "gt.dxf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("bench/cases"))
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    levels = ("clean", "scan", "rough")
    for i in range(args.count):
        level = levels[i % len(levels)]
        name = f"synth_{i:03d}_{level}"
        make_case(args.out / name, args.seed * 1000 + i, level)
        print("wrote", args.out / name)
    (args.out / "README.txt").write_text("Generated by `python -m bench.synth`. Safe to delete and regenerate.\n")


if __name__ == "__main__":
    main()
