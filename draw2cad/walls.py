"""Find walls in a binarised plan and turn them into clean, axis-aligned wall bands (pixels).

Works for the three common wall styles:
- filled (solid black): thick strokes survive a morphological opening that removes thin lines;
- outlined (two face lines) and hatched: face pairs are bridged by a directional closing whose
  length comes from the dominant spacing between parallel lines, then treated like filled walls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from draw2cad.raster import Word


@dataclass
class Band:
    """An axis-aligned wall piece. For horizontal bands `lo..hi` are x and `f0..f1` are y (px)."""

    horizontal: bool
    lo: float
    hi: float
    f0: float
    f1: float
    exterior: bool = False
    outer: int = 0  # which face is the outside: -1 = f0, +1 = f1 (exterior walls)

    @property
    def t(self) -> float:
        return self.f1 - self.f0

    @property
    def length(self) -> float:
        return self.hi - self.lo

    @property
    def mid(self) -> float:
        return (self.f0 + self.f1) / 2

    def rect(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) in image pixels."""
        if self.horizontal:
            return self.lo, self.f0, self.hi, self.f1
        return self.f0, self.lo, self.f1, self.hi


@dataclass
class WallResult:
    bands: list[Band]
    clean: np.ndarray  # ink with text removed
    mask: np.ndarray  # wall pixels before vectorising
    pen: float
    t_max: float  # thickest plausible wall, px
    style: str
    footprint: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.uint8))


def _ellipse(k: int) -> np.ndarray:
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def remove_text(ink: np.ndarray, words: list[Word], pen: float) -> np.ndarray:
    """Drop small connected components (letters, symbols) and anything inside an OCR text box."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    H, W = ink.shape
    small = max(10 * pen, 0.012 * max(H, W))
    x, y, w, h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    drop = (w < small) & (h < small)
    for wd in words:
        bx0, by0, bx1, by1 = wd.box
        mx, my = 0.15 * (bx1 - bx0) + 2, 0.15 * (by1 - by0) + 2
        inside = (x >= bx0 - mx) & (y >= by0 - my) & (x + w <= bx1 + mx) & (y + h <= by1 + my)
        drop |= inside
    drop[0] = True
    return np.where(drop[lab], 0, ink).astype(np.uint8)


def parallel_spacing(clean: np.ndarray, pen: float) -> list[tuple[int, float]]:
    """Peaks of the spacing between parallel long lines: (centre-to-centre distance px, strength)."""
    H, W = clean.shape
    k = max(15, int(0.012 * max(H, W)))
    V = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))) > 0
    Hl = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))) > 0
    g_lo, g_hi = int(2 * pen) + 2, int(0.05 * max(H, W))
    score = np.zeros(g_hi + 1)
    for g in range(g_lo, g_hi + 1):
        score[g] = np.count_nonzero(V[:, :-g] & V[:, g:]) + np.count_nonzero(Hl[:-g] & Hl[g:])
    s = np.convolve(score, np.ones(3) / 3, mode="same")
    top = s[g_lo:].max() if g_hi > g_lo else 0
    peaks = []
    for g in range(g_lo + 1, g_hi):
        if s[g] >= s[g - 1] and s[g] > s[g + 1] and s[g] >= 0.2 * top and top > 0:
            peaks.append((g, float(s[g] / top)))
    return peaks


def tone_fill(gray: np.ndarray, ink: np.ndarray, pen: float) -> np.ndarray:
    """Areas filled with pale grey (light poché, or hatching blurred together by scanning)."""
    # Ink is handled by the line cues; blank it out so that smoothing does not turn every dark
    # line into "grey area". Wide smoothing also dilutes thin pale lines below the threshold,
    # while a filled or densely hatched wall stays uniformly darker than the paper.
    paper = float(np.median(gray))
    gray = np.where(ink > 0, paper, gray).astype(np.uint8)
    k = int(max(5, 4 * pen + 1)) | 1
    smooth = cv2.blur(gray, (k, k)).astype(np.float32)
    noise = 1.4826 * float(np.median(np.abs(smooth[smooth > paper - 15] - paper)))
    fill = np.where(smooth < paper - max(15.0, 4 * noise), 255, 0).astype(np.uint8)
    return cv2.morphologyEx(fill, cv2.MORPH_OPEN, _ellipse(k))


def narrow_enclosed(clean: np.ndarray, pen: float, max_width: float) -> np.ndarray:
    """White regions fully enclosed by lines and nowhere wider than a wall.

    In an outlined plan every wall is such a region (between its faces, closed by corners and
    jamb lines), whatever its shape: steps, corners and short stubs included. Rooms are wider.
    """
    from scipy import ndimage

    lines = cv2.dilate(clean, np.ones((3, 3), np.uint8))
    free = (lines == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    if n <= 1:
        return np.zeros_like(clean)
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    idx = np.arange(1, n)
    widest = ndimage.maximum(dt, lab, idx) * 2
    area = np.bincount(lab.ravel(), minlength=n)[1:]
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    ok = (widest <= max_width) & (area >= (3 * pen) ** 2)
    ok[border[border > 0] - 1] = False
    keep = np.zeros(n, bool)
    keep[1:] = ok
    fill = keep[lab].astype(np.uint8) * 255
    return cv2.dilate(fill, np.ones((3, 3), np.uint8), iterations=2)  # take the face lines along


ALL_CUES = ("solid", "tone", "pairs", "narrow")


def wall_mask(clean: np.ndarray, gray: np.ndarray, pen: float, max_gap: float,
              cues: tuple[str, ...] = ALL_CUES) -> tuple[np.ndarray, float, str]:
    """Cues: solid = thick strokes; tone = pale grey fill; pairs = bridge parallel face lines;
    narrow = fill enclosed spaces no wider than a wall."""
    k0 = int(round(2 * pen + 1))
    solid = cv2.morphologyEx(clean, cv2.MORPH_OPEN, _ellipse(k0)) if "solid" in cues else np.zeros_like(clean)
    if "tone" in cues:
        solid = cv2.bitwise_or(solid, tone_fill(gray, clean, pen))
    filled = np.count_nonzero(solid) > 0.25 * max(1, np.count_nonzero(clean))

    # The thickest wall's face spacing: the widest strong peak that is still plausible for a wall.
    # Wider peaks come from furniture, title-block rows and roof hatching.
    peaks = [(g, s) for g, s in parallel_spacing(clean, pen) if g <= max_gap]
    gap = 0.0
    if peaks:
        top = max(s for _, s in peaks)
        gap = max(g for g, s in peaks if s >= 0.3 * top)
    bridged = np.zeros_like(clean)
    if gap > 0 and "pairs" in cues:
        L = int(gap + pen + 2)
        ch = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        cv = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        bridged = cv2.morphologyEx(cv2.bitwise_or(ch, cv), cv2.MORPH_OPEN, _ellipse(k0 + 2))
    mask = cv2.bitwise_or(solid, bridged)
    if "narrow" in cues:
        narrow = narrow_enclosed(clean, pen, max_gap + pen)
        mask = cv2.bitwise_or(mask, cv2.morphologyEx(narrow, cv2.MORPH_OPEN, _ellipse(k0 + 2)))

    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    t_solid = 2 * float(np.percentile(dt[dt > 0], 99)) + 1 if np.count_nonzero(dt) else 0.0
    t_max = max(t_solid, gap + pen if gap else 0.0, 3 * pen)
    style = "filled" if filled else ("outlined" if gap else "unknown")
    return mask, t_max, style


def _bands_one_way(mask: np.ndarray, horizontal: bool, min_len: int, t_max: float) -> list[Band]:
    m = mask if horizontal else mask.T
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    out = []
    tol = max(2, int(0.15 * t_max))
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        sub = lab[y:y + h, x:x + w] == i
        cols = sub.any(0)
        if cols.sum() < min_len:
            continue
        xs = np.flatnonzero(cols)
        top = np.argmax(sub, axis=0)[xs]
        bot = (h - 1 - np.argmax(sub[::-1], axis=0))[xs]
        # Split into runs of steady faces: a wall with a wardrobe or a bathtub drawn against it
        # forms one blob, but only the stretch of constant thickness is wall.
        start = 0
        for j in range(1, len(xs) + 1):
            if j < len(xs) and xs[j] == xs[j - 1] + 1 and abs(int(top[j]) - int(top[j - 1])) <= tol \
                    and abs(int(bot[j]) - int(bot[j - 1])) <= tol:
                continue
            if j - start >= min_len:
                f0 = y + float(np.median(top[start:j]))
                f1 = y + float(np.median(bot[start:j])) + 1
                if 2 <= f1 - f0 <= 1.3 * t_max:
                    out.append(Band(horizontal, float(x + xs[start]), float(x + xs[j - 1] + 1), f0, f1))
            start = j
    return out


def _cluster(values: list[float], weights: list[float], tol: float) -> dict[int, float]:
    """Map each index to the weighted mean of its cluster (1-D, gap-based)."""
    order = np.argsort(values)
    out: dict[int, float] = {}
    group: list[int] = []

    def flush():
        if group:
            w = np.array([weights[i] for i in group])
            v = np.array([values[i] for i in group])
            m = float((v * w).sum() / w.sum())
            for i in group:
                out[i] = m

    for idx in order:
        if group and values[idx] - values[group[-1]] > tol:
            flush()
            group = []
        group.append(int(idx))
    flush()
    return out


def regularize(bands: list[Band], tol: float) -> list[Band]:
    """Make faces of the same line exactly equal, and make wall ends meet perpendicular faces."""
    for horizontal in (True, False):
        bs = [b for b in bands if b.horizontal == horizontal]
        if not bs:
            continue
        vals = [b.f0 for b in bs] + [b.f1 for b in bs]
        wts = [b.length for b in bs] * 2
        m = _cluster(vals, wts, tol)
        for i, b in enumerate(bs):
            b.f0, b.f1 = m[i], m[i + len(bs)]
    for b in bands:
        faces = sorted({f for o in bands if o.horizontal != b.horizontal for f in (o.f0, o.f1)})
        if not faces:
            continue
        fa = np.array(faces)
        for attr in ("lo", "hi"):
            v = getattr(b, attr)
            j = int(np.argmin(np.abs(fa - v)))
            if abs(fa[j] - v) <= 3 * tol:
                setattr(b, attr, float(fa[j]))
    return [b for b in bands if b.length > b.t]


def merge_collinear(bands: list[Band], tol: float) -> list[Band]:
    """Join pieces of the same wall line that touch or overlap."""
    out: list[Band] = []
    key = lambda b: (b.horizontal, round(b.f0 / tol), round(b.f1 / tol), b.lo)
    for b in sorted(bands, key=key):
        p = out[-1] if out else None
        if (p and p.horizontal == b.horizontal and abs(p.f0 - b.f0) <= tol and abs(p.f1 - b.f1) <= tol
                and b.lo <= p.hi + 1):
            p.hi = max(p.hi, b.hi)
        else:
            out.append(Band(b.horizontal, b.lo, b.hi, b.f0, b.f1))
    return out


def footprint_mask(shape: tuple[int, int], bands: list[Band], bridge: int,
                   room_points: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Filled building outline: walls with door/window gaps bridged, holes filled.

    With `room_points` (positions of recognised room names), only outlines that contain a room
    name count as buildings; that rules out title blocks and legend frames, which also enclose space.
    """
    m = np.zeros(shape, np.uint8)
    for b in bands:
        x0, y0, x1, y1 = b.rect()
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), 255, -1)
    walls = m.copy()
    k = max(3, bridge)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        one = np.zeros_like(m)
        cv2.drawContours(one, [c], -1, 255, -1)
        # A building encloses rooms; a shape that is mostly wall (a margin bar, a title-block
        # rule, a solid legend symbol) encloses nothing and is not a building.
        if np.count_nonzero(one) < 2 * np.count_nonzero(walls & one):
            continue
        rooms = sum(1 for x, y in room_points or [] if cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0)
        kept.append((rooms, np.count_nonzero(one), one))
    filled = np.zeros_like(m)
    if not kept:
        return filled
    if any(r for r, _, _ in kept):
        for r, _, one in kept:
            if r:
                filled |= one
    else:
        biggest = max(a for _, a, _ in kept)
        for _, a, one in kept:
            if a >= 0.25 * biggest:
                filled |= one
    return filled


def classify_exterior(bands: list[Band], footprint: np.ndarray, pen: float) -> None:
    H, W = footprint.shape
    off = int(max(2, 1.5 * pen))
    for b in bands:
        fracs = []
        for side, face in ((-1, b.f0 - off), (1, b.f1 + off)):
            n = max(8, int(b.length / 4))
            along = np.linspace(b.lo + b.t, b.hi - b.t, n) if b.length > 3 * b.t else np.linspace(b.lo, b.hi, n)
            if b.horizontal:
                xs, ys = along, np.full(n, face)
            else:
                xs, ys = np.full(n, face), along
            xi = np.clip(xs.astype(int), 0, W - 1)
            yi = np.clip(ys.astype(int), 0, H - 1)
            outside = (footprint[yi, xi] == 0) | (xs < 0) | (ys < 0) | (xs >= W) | (ys >= H)
            fracs.append((outside.mean(), side))
        frac, side = max(fracs)
        if frac >= 0.5:
            b.exterior, b.outer = True, side


def extract_walls(ink: np.ndarray, gray: np.ndarray, words: list[Word], pen: float,
                  max_wall_px: float | None = None, clean: np.ndarray | None = None,
                  door_px: float | None = None, room_points: list[tuple[float, float]] | None = None,
                  cues: tuple[str, ...] = ALL_CUES) -> WallResult:
    """`max_wall_px`: thickest plausible wall in px, known once the scale is (e.g. 520 mm / scale).
    `door_px`: widest door/window gap to bridge when finding the building outline (scale known)."""
    if clean is None:
        clean = remove_text(ink, words, pen)
    gray = gray.copy()
    for w in words:  # text boxes must not count as grey fill either
        x0, y0, x1, y1 = (int(v) for v in w.box)
        gray[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 255
    if max_wall_px:
        max_gap = max_wall_px
    else:
        # Without a scale: walls are rarely thicker than ~1.5 text heights at 1:100.
        heights = [w.size for w in words if len(w.text) >= 2]
        max_gap = 1.5 * float(np.median(heights)) if len(heights) >= 3 else 0.02 * max(ink.shape)
    mask, t_max, style = wall_mask(clean, gray, pen, max_gap, tuple(cues))
    t_max = min(t_max, 1.15 * max_gap + pen)
    min_len = int(max(1.2 * t_max, 6 * pen))
    bands = _bands_one_way(mask, True, min_len, t_max) + _bands_one_way(mask, False, min_len, t_max)
    tol = max(1.5, 0.75 * pen)
    bands = regularize(bands, tol)
    bands = merge_collinear(bands, tol)
    # Building outline: walls with openings bridged, kept only where it encloses room names (or,
    # without names, the larger outlines). Thin lines are left out so a dimension line or the sheet
    # frame cannot enlarge it. Anything outside the outline is not a wall.
    # Start with the thickest walls (exterior walls are the thickest); loosen only if they do not
    # form a building outline, since thin bands also include leftover dimension lines.
    fp = np.zeros(ink.shape, np.uint8)
    for frac in (0.5, 0.35, 0.2):
        thick = [b for b in bands if b.t >= max(3 * pen, frac * t_max)] or bands
        fp = footprint_mask(ink.shape, thick, int(door_px or 8 * t_max), room_points)
        if np.count_nonzero(fp):
            break
    near = cv2.dilate(fp, _ellipse(int(t_max)))
    bands = [b for b in bands if _inside_fraction(b, near) >= 0.5]
    classify_exterior(bands, fp, pen)
    return WallResult(bands, clean, mask, pen, t_max, style, fp)


def _inside_fraction(b: Band, region: np.ndarray) -> float:
    x0, y0, x1, y1 = (int(v) for v in b.rect())
    sub = region[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    return float(np.count_nonzero(sub)) / max(1, sub.size)
