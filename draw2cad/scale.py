"""Working out mm per pixel, from as many independent clues as the drawing offers.

Order of trust:
1. the user's own calibration;
2. written dimensions: first measured against their dimension lines (needs no walls), then
   refined against the reconstructed building outline (outer faces, per axis);
3. a scale note ("SKALA 1:100") combined with the paper size or resolution;
4. typical Swedish wall thickness / door widths (rough, last resort).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from draw2cad.ocr import find_paper_notes, find_scale_notes, parse_mm
from draw2cad.raster import Page, Word

PAPER_MM = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (297, 210)}


@dataclass
class Scale:
    sx: float  # mm per px along x
    sy: float  # mm per px along y
    source: str
    confidence: float
    notes: list[str] = field(default_factory=list)
    # Lines that carry written dimensions: (horizontal, position across, start, end) in px.
    dim_lines: list[tuple[bool, float, float, float]] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return (self.sx + self.sy) / 2


def _long_lines(clean: np.ndarray, horizontal: bool, min_len: int) -> list[tuple[float, float, float]]:
    """Long straight lines as (position across, start, end) in px."""
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1) if horizontal else (1, 7))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1) if horizontal else (1, min_len))
    m = cv2.morphologyEx(cv2.morphologyEx(clean, cv2.MORPH_CLOSE, bridge), cv2.MORPH_OPEN, k)
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    pieces = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if horizontal:
            pieces.append((float(cent[i][1]), float(x), float(x + w)))
        else:
            pieces.append((float(cent[i][0]), float(y), float(y + h)))
    # Faint lines on scans break into pieces (noise, text boxes cut out); join collinear pieces.
    # Two separate dimension lines at exactly the same height are rare on a plan.
    max_gap = 0.25 * max(clean.shape)
    out: list[tuple[float, float, float]] = []
    for pos, a, b in sorted(pieces, key=lambda p: (round(p[0] / 3), p[1])):
        if out and abs(out[-1][0] - pos) <= 3 and a - out[-1][2] <= max_gap:
            p0, a0, b0 = out[-1]
            out[-1] = (p0, a0, max(b0, b))
        else:
            out.append((pos, a, b))
    return out


def from_dimension_lines(words: list[Word], clean: np.ndarray, found: list | None = None,
                         support: list | None = None) -> tuple[float | None, float | None, list[str]]:
    """mm/px per axis from written dimensions and the dimension lines they sit on.

    A dimension line either carries one overall value, or is a chain (3 300 + 5 200 + 4 900 on one
    continuous line) whose total is the line length. Both readings are tried; the reading that
    most lines agree on wins.
    """
    H, W = clean.shape
    notes = []
    result = []
    for horizontal in (True, False):
        lines = _long_lines(clean, horizontal, int(0.015 * max(H, W)))
        per_line: dict[int, list[int]] = {}
        for w in words:
            v = parse_mm(w.text)
            if v is None or v < 1000 or w.vertical == horizontal:
                continue
            cx, cy = w.center
            along, across = (cx, cy) if horizontal else (cy, cx)
            size = w.size
            best, best_d = None, None
            for i, (pos, a, b) in enumerate(lines):
                d = abs(pos - across)
                if a - size <= along <= b + size and d <= 1.6 * size and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best is not None:
                per_line.setdefault(best, []).append(v)
        if found is not None:
            found += [(horizontal, *lines[i]) for i in per_line]
        cands: list[tuple[float, int, int]] = []  # (mm/px, line, value)
        for i, vals in per_line.items():
            _, a, b = lines[i]
            for v in {max(vals), sum(vals)}:
                cands.append((v / (b - a), i, v))
        if cands:
            # Each line votes for every candidate within 3%; the best-supported value wins,
            # ties going to the larger dimension (relatively more precise).
            def votes(c):
                return len({i for r, i, _ in cands if abs(r / c[0] - 1) < 0.03}), c[2]
            best = max(cands, key=votes)
            n_lines = votes(best)[0]
            agree = [r for r, _, _ in cands if abs(r / best[0] - 1) < 0.03]
            r = float(np.median(agree))
            notes.append(f"{'horizontal' if horizontal else 'vertical'} dimension lines: "
                         f"{n_lines} of {len(per_line)} agree on {r:.3f} mm/px")
            result.append(r)
            if support is not None:
                support.append(n_lines)
        else:
            result.append(None)
            if support is not None:
                support.append(0)
    return result[0], result[1], notes


def from_scale_note(page: Page, words: list[Word], exact: list | None = None) -> tuple[float | None, list[str]]:
    """`exact` gets True appended when the pixel density is known (PDF page size, real DPI)."""
    scales = find_scale_notes(words)
    if exact is not None:
        exact.append(page.dpi is not None)
    if not scales:
        return None, []
    scale = max(set(scales), key=scales.count)
    H, W = page.gray.shape
    dpi = page.dpi
    how = "known resolution"
    if dpi is None:
        papers = find_paper_notes(words)
        names = [papers[0]] if papers else []
        # Otherwise: an A-series sheet scanned whole has aspect ratio sqrt(2).
        if not names and abs(max(H, W) / min(H, W) - 2 ** 0.5) < 0.03:
            names = ["A3"]  # most common for house plans; checked against priors by the caller
        if not names:
            return None, [f"scale note 1:{scale} found, but paper size unknown"]
        long_mm = PAPER_MM[names[0]][0]
        dpi = max(H, W) / (long_mm / 25.4)
        how = f"paper {names[0]}" + ("" if papers else " (assumed from sheet shape)")
    s = 25.4 / dpi * scale
    return s, [f"scale note 1:{scale}, {how}: {s:.3f} mm/px"]


def refine_with_outline(scale: Scale, words: list[Word], outline_px: list[tuple[float, float, float, float]]) -> Scale:
    """Use overall dimensions against outer wall faces of each building (more exact than dim lines)."""
    if not outline_px:
        return scale
    sx_list, sy_list = [], []
    for x0, y0, x1, y1 in outline_px:
        Wp, Dp = x1 - x0, y1 - y0
        for w in words:
            v = parse_mm(w.text)
            if v is None or v < 1000:
                continue
            cx, cy = w.center
            if not w.vertical and x0 - 0.1 * Wp <= cx <= x1 + 0.1 * Wp and (y0 - 0.5 * Dp < cy < y0 or y1 < cy < y1 + 0.5 * Dp):
                cand = v / Wp
                if abs(cand / scale.sx - 1) < 0.1:
                    sx_list.append(cand)
            if w.vertical and y0 - 0.1 * Dp <= cy <= y1 + 0.1 * Dp and (x0 - 0.5 * Wp < cx < x0 or x1 < cx < x1 + 0.5 * Wp):
                cand = v / Dp
                if abs(cand / scale.sy - 1) < 0.1:
                    sy_list.append(cand)
    notes = list(scale.notes)
    sx, sy = scale.sx, scale.sy
    if sx_list:
        sx = max(sx_list, key=lambda c: -abs(c / scale.sx - 1))  # closest overall width
        notes.append(f"width refined from overall dimension to outer faces: {sx:.4f} mm/px")
    if sy_list:
        sy = max(sy_list, key=lambda c: -abs(c / scale.sy - 1))
        notes.append(f"depth refined from overall dimension to outer faces: {sy:.4f} mm/px")
    if sx_list and not sy_list and abs(sx / scale.sy - 1) < 0.02:
        sy = sx * scale.sy / scale.sx
    if sy_list and not sx_list and abs(sy / scale.sx - 1) < 0.02:
        sx = sy * scale.sx / scale.sy
    src = scale.source + (" + outline" if sx_list or sy_list else "")
    return Scale(sx, sy, src, min(0.98, scale.confidence + (0.1 if sx_list or sy_list else 0)), notes,
                 scale.dim_lines)


def plausible(s: float | None, words: list[Word]) -> bool:
    """Lettering on plans is ~2.5-3.5 mm at 1:50-1:100, i.e. ~100-600 mm in real-world size."""
    if not s:
        return False
    heights = [w.size for w in words if len(w.text) >= 2]
    if len(heights) < 3:
        return True
    return 80 <= float(np.median(heights)) * s <= 700


def initial_scale(page: Page, words: list[Word], clean: np.ndarray, user: dict | None = None) -> Scale | None:
    dim_lines: list = []  # found either way: wall detection needs them erased
    support: list[int] = []
    sx, sy, notes = from_dimension_lines(words, clean, dim_lines, support)
    if user:
        if "mm_per_px" in user:
            s = float(user["mm_per_px"])
            return Scale(s, s, "user", 1.0, [f"set by user: {s:.4f} mm/px"], dim_lines)
        if "scale" in user and "dpi" in user:
            s = 25.4 / float(user["dpi"]) * float(user["scale"])
            return Scale(s, s, "user", 1.0, [f"user: 1:{user['scale']} at {user['dpi']} dpi"], dim_lines)
    for name, v in (("horizontal", sx), ("vertical", sy)):
        if v and not plausible(v, words):
            notes.append(f"{name} dimension reading {v:.3f} mm/px rejected (text would be implausibly sized)")
    sx = sx if plausible(sx, words) else None
    sy = sy if plausible(sy, words) else None
    if sx and sy and abs(sx / sy - 1) > 0.05:  # axes disagree: keep the better-supported one
        notes.append("horizontal and vertical readings disagree; using horizontal")
        sy = sx
    exact: list[bool] = []
    note_s, note_notes = from_scale_note(page, words, exact)
    if note_s and not plausible(note_s, words):
        note_notes.append("scale note rejected (paper size assumption does not fit)")
        note_s = None
    notes += note_notes
    note_exact = bool(note_s and exact and exact[0])
    if note_exact and (sx or sy):
        dims = (sx or sy, sy or sx)
        if abs((dims[0] + dims[1]) / 2 / note_s - 1) > 0.05 and max(support or [0]) < 2:
            # One stray number on one line (an order number in the title block, say) must not
            # outvote a scale note on a sheet whose size is known exactly.
            notes.append("dimension reading disagrees with the exact scale note and has only one line "
                         "behind it; using the scale note")
            sx = sy = None
            dim_lines = []
    if note_exact and not (sx or sy):
        return Scale(note_s, note_s, "scale note + known sheet size", 0.9, notes, dim_lines)
    if sx or sy:
        sx, sy = sx or sy, sy or sx
        conf = 0.85
        if note_s and abs((sx + sy) / 2 / note_s - 1) < 0.05:
            conf = 0.92
            notes.append("scale note agrees with dimensions")
        return Scale(sx, sy, "dimensions", conf, notes, dim_lines)
    if note_s:
        return Scale(note_s, note_s, "scale note", 0.6, notes, dim_lines)
    return None


def prior_scale(t_ext_px: float | None, door_px: list[float], notes: list[str]) -> Scale | None:
    """Last resort: typical Swedish exterior wall (~350 mm) and door openings (~910 mm)."""
    if door_px and len(door_px) >= 2:
        s = 910 / float(np.median(door_px))
        return Scale(s, s, "typical door width", 0.35, notes + [f"assumed doors ~910 mm: {s:.3f} mm/px"])
    if t_ext_px:
        s = 350 / t_ext_px
        return Scale(s, s, "typical wall thickness", 0.25, notes + [f"assumed exterior walls ~350 mm: {s:.3f} mm/px"])
    return None
