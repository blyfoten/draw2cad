"""Doors, windows and plain openings, found along the wall bands (pixels)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from draw2cad.walls import Band


@dataclass
class OpenPx:
    kind: str  # "door" | "window" | "opening"
    c0: float  # along-wall start (px)
    c1: float
    side: int = 0  # doors: which side the leaf swings to, -1 = towards f0, +1 = towards f1
    hinge_lo: bool = True  # doors: hinge at the c0 end


@dataclass
class WallPx:
    band: Band
    openings: list[OpenPx] = field(default_factory=list)


def _crop(img: np.ndarray, b: Band, a0: float, a1: float, f0: float, f1: float) -> np.ndarray:
    """Sub-image with rows = across the wall (f), columns = along the wall (a)."""
    H, W = img.shape
    if b.horizontal:
        sub = img[max(0, int(f0)):max(0, int(f1)), max(0, int(a0)):max(0, int(a1))]
        return sub
    sub = img[max(0, int(a0)):max(0, int(a1)), max(0, int(f0)):max(0, int(f1))]
    return sub.T


def _runs_per_column(sub: np.ndarray) -> np.ndarray:
    b = (sub > 0).astype(np.int8)
    if b.size == 0:
        return np.zeros(0, int)
    d = np.diff(np.pad(b, ((1, 0), (0, 0))), axis=0)
    return (d == 1).sum(axis=0)


def _parallel_lines(img: np.ndarray, b: Band, a0: float, a1: float, pen: float) -> np.ndarray:
    """Ink across the wall at a0..a1, keeping only strokes that run along the wall.

    The filter length follows the wall thickness: long enough to remove hatching and cross lines,
    short enough to keep the glass lines of a narrow window."""
    sub = _crop(img, b, a0, a1, b.f0 - 2 * pen, b.f1 + 2 * pen)
    if sub.size == 0:
        return sub
    k = max(3, int(min(0.3 * (a1 - a0), 1.2 * b.t)))
    return cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))


def classify_gap(b: Band, g0: float, g1: float, clean: np.ndarray, thin: np.ndarray, pen: float) -> OpenPx:
    w = g1 - g0
    # Window (exterior walls only): two or more lines run along the wall through the gap.
    par = _parallel_lines(clean, b, g0 + 2 * pen, g1 - 2 * pen, pen)
    if b.exterior and par.size and (_runs_per_column(par) >= 2).mean() >= 0.6:
        return OpenPx("window", g0, g1)
    # Door: a swing arc and leaf on one side, filling a w x w square with thin ink.
    best = None
    expected = 2.57 * pen / max(w, 1)  # (pi/2 + 1) * w * pen / w^2
    for side, (f0, f1) in ((-1, (b.f0 - w, b.f0 - pen)), (1, (b.f1 + pen, b.f1 + w))):
        sq = _crop(thin, b, g0, g1, f0, f1)
        if sq.size == 0:
            continue
        density = np.count_nonzero(sq) / sq.size
        if 0.3 * expected <= density <= 6 * expected and (best is None or density > best[0]):
            # The leaf runs across the whole square at the hinge end; the arc only meets the
            # wall at the other end.
            strip = max(2, int(3 * pen))
            lo = np.count_nonzero(sq[:, :strip])
            hi = np.count_nonzero(sq[:, -strip:])
            best = (density, side, lo >= hi)
    if best:
        return OpenPx("door", g0, g1, side=best[1], hinge_lo=best[2])
    return OpenPx("opening", g0, g1)


def join_walls(bands: list[Band], clean: np.ndarray, thin: np.ndarray, pen: float, t_max: float) -> list[WallPx]:
    """Merge collinear bands separated by a door/window-sized gap into one wall with an opening."""
    tol = max(1.5, 0.75 * pen)
    max_gap = max(8 * t_max, 40 * pen)
    walls: list[WallPx] = []
    for b in sorted(bands, key=lambda b: (b.horizontal, round(b.mid / tol), b.lo)):
        prev = walls[-1] if walls else None
        if prev and prev.band.horizontal == b.horizontal and abs(prev.band.f0 - b.f0) <= tol \
                and abs(prev.band.f1 - b.f1) <= tol and prev.band.exterior == b.exterior:
            gap = b.lo - prev.band.hi
            if gap <= max(2.0, 0.6 * b.t):
                prev.band.hi = max(prev.band.hi, b.hi)
                continue
            if gap <= max_gap:
                prev.openings.append(classify_gap(prev.band, prev.band.hi, b.lo, clean, thin, pen))
                prev.band.hi = b.hi
                continue
        walls.append(WallPx(Band(b.horizontal, b.lo, b.hi, b.f0, b.f1, b.exterior, b.outer)))
    return walls


def doors_at_ends(walls: list[WallPx], clean: np.ndarray, thin: np.ndarray, pen: float, max_gap: float) -> None:
    """A door often sits between a wall's end and a perpendicular wall (the stub on the far side
    is too short to be a wall piece). If a door swing is there, extend the wall to meet it."""
    for wp in walls:
        b = wp.band
        if b.length < 3 * b.t:
            continue  # door jambs and short stubs are not walls to extend
        for end in ("lo", "hi"):
            if any((o.c1 >= b.hi - 1) if end == "hi" else (o.c0 <= b.lo + 1) for o in wp.openings):
                continue
            best = None
            for other in walls:
                o = other.band
                if o.horizontal == b.horizontal or not (o.lo - b.t <= b.mid <= o.hi + b.t):
                    continue
                d = (o.f0 - b.hi) if end == "hi" else (b.lo - o.f1)
                if 4 * pen < d <= max_gap and (best is None or d < best[0]):
                    best = (d, o)
            if best is None:
                continue
            d, o = best
            g0, g1 = (b.hi, o.f0) if end == "hi" else (o.f1, b.lo)
            op = classify_gap(b, g0, g1, clean, thin, pen)
            if op.kind == "door":
                wp.openings.append(op)
                if end == "hi":
                    b.hi = o.mid
                else:
                    b.lo = o.mid
        wp.openings.sort(key=lambda o: o.c0)


def windows_inside(walls: list[WallPx], clean: np.ndarray, pen: float) -> None:
    """Windows drawn inside outlined/hatched exterior walls (no gap): glass lines along the middle."""
    for wp in walls:
        b = wp.band
        a0, a1 = b.lo + b.t, b.hi - b.t
        if not b.exterior or a1 - a0 < 2 * b.t or b.t < 4 * pen:
            continue
        par = _parallel_lines(clean, b, a0, a1, pen)
        if par.size == 0:
            continue
        # An outlined wall is empty between its faces (hatching is removed by the along-wall
        # opening); a window has glass line(s) there. Filled walls are solid and excluded.
        off = int(2 * pen)
        mid = par[off + int(0.25 * b.t):off + int(0.75 * b.t) + 1]
        solid = (par > 0).mean(axis=0) > 0.8
        is_win = (mid > 0).any(axis=0) & ~solid
        # Close small breaks, then take stretches at least ~a wall thickness long.
        k = max(1, int(2 * pen))
        is_win = cv2.morphologyEx(is_win.astype(np.uint8)[None], cv2.MORPH_CLOSE,
                                  np.ones((1, 2 * k + 1), np.uint8))[0] > 0
        idx = np.flatnonzero(np.diff(np.concatenate([[0], is_win.astype(np.int8), [0]])))
        for s, e in zip(idx[::2], idx[1::2]):
            if e - s >= max(0.9 * b.t, 6 * pen):
                c0, c1 = a0 + s, a0 + e
                if not any(o.c0 - pen < c1 and c0 < o.c1 + pen for o in wp.openings):
                    wp.openings.append(OpenPx("window", c0, c1))
        wp.openings.sort(key=lambda o: o.c0)


def find_openings(bands: list[Band], clean: np.ndarray, wall_mask: np.ndarray, pen: float, t_max: float,
                  window_img: np.ndarray | None = None, max_door_px: float | None = None) -> list[WallPx]:
    """`window_img`: where to look for window glass lines, if not in `clean` (vector PDFs draw
    windows with a different pen than the walls). `max_door_px`: widest door, when the scale is known."""
    walls_img = np.zeros_like(clean)
    for b in bands:
        x0, y0, x1, y1 = b.rect()
        cv2.rectangle(walls_img, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    thin = cv2.bitwise_and(clean, cv2.bitwise_not(cv2.dilate(walls_img, np.ones((3, 3), np.uint8), iterations=int(pen))))
    walls = join_walls(bands, clean if window_img is None else window_img, thin, pen, t_max)
    doors_at_ends(walls, clean, thin, pen, max_door_px or max(3 * t_max, 20 * pen))
    windows_inside(walls, clean if window_img is None else window_img, pen)
    return walls
