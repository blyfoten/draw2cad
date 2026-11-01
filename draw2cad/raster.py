"""Loading PDFs and images into a clean, deskewed, binarised page."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 5000  # px; larger inputs are downscaled (detail beyond this does not help)
PDF_DPI = 300


@dataclass
class Word:
    """A piece of text with its pixel box (x0, y0, x1, y1) in the page image."""

    text: str
    box: tuple[float, float, float, float]
    conf: float = 1.0
    vertical: bool = False  # reads bottom-to-top (vertical dimension text)

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def size(self) -> float:
        """Text height in px (across the reading direction)."""
        w, h = self.box[2] - self.box[0], self.box[3] - self.box[1]
        return w if self.vertical else h


@dataclass
class Page:
    gray: np.ndarray  # uint8, white paper, dark ink, deskewed
    ink: np.ndarray  # uint8 0/255, 255 = ink
    dpi: float | None  # known pixel density (PDF render or trustworthy image metadata)
    paper_mm: tuple[float, float] | None  # known paper size (PDFs)
    words: list[Word] = field(default_factory=list)  # embedded text (vector PDFs)
    skew_deg: float = 0.0
    source: str = "image"
    pages: int = 1
    # Vector PDFs: stroked linework as (pen width in pt, N x 2 points in page px).
    vectors: list[tuple[float, np.ndarray]] = field(default_factory=list)


def load(data: bytes, filename: str, page: int = 0) -> Page:
    vectors: list[tuple[float, np.ndarray]] = []
    if data[:5] == b"%PDF-" or filename.lower().endswith(".pdf"):
        gray, dpi, paper, words, pages, vectors = _load_pdf(data, page)
        source = "pdf"
    else:
        gray, dpi = _load_image(data)
        paper, words, pages, source = None, [], 1, "image"
    scale = min(1.0, MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dpi = dpi * scale if dpi else None
        words = [Word(w.text, tuple(v * scale for v in w.box), w.conf, w.vertical) for w in words]
        vectors = [(w, pts * scale) for w, pts in vectors]
    gray = normalize_background(gray)
    skew = estimate_skew(gray)
    if abs(skew) > 0.05:
        gray, words = rotate(gray, words, skew)
        vectors = []  # a rotated vector PDF is rare; fall back to the raster rather than transform
    return Page(gray=gray, ink=binarize(gray), dpi=dpi, paper_mm=paper, words=words, skew_deg=skew,
                source=source, pages=pages, vectors=vectors)


def _load_pdf(data: bytes, index: int):
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    page = doc[min(index, len(doc) - 1)]
    w_mm, h_mm = page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72
    dpi = min(PDF_DPI, MAX_SIDE / (max(w_mm, h_mm) / 25.4))
    pix = page.get_pixmap(dpi=int(dpi), colorspace=pymupdf.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    real_dpi = pix.width / (w_mm / 25.4)
    k = real_dpi / 72
    words = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        vertical = (y1 - y0) > 1.5 * (x1 - x0) and len(text) > 1
        words.append(Word(text, (x0 * k, y0 * k, x1 * k, y1 * k), 1.0, vertical))
    return gray, real_dpi, (w_mm, h_mm), merge_number_words(words), len(doc), _vector_lines(page, k)


def _vector_lines(page, k: float) -> list[tuple[float, np.ndarray]]:
    """Paths of a PDF page as polylines in px, tagged with their pen width (pt).

    Fill-only paths (window glass, poché) are tagged -1 so they never mix with a stroke pen."""
    t = np.linspace(0, 1, 12)[:, None]
    out = []
    for path in page.get_drawings():
        kind = path.get("type") or ""
        width = round(float(path.get("width") or 0), 2) if "s" in kind else -1.0
        for it in path["items"]:
            op = it[0]
            if op == "l":
                pts = np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]])
            elif op == "c":  # cubic Bezier
                p0, p1, p2, p3 = (np.array([p.x, p.y]) for p in it[1:5])
                pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
            elif op == "re":
                r = it[1]
                pts = np.array([[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1], [r.x0, r.y0]])
            elif op == "qu":
                q = it[1]
                pts = np.array([[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y], [q.ul.x, q.ul.y]])
            else:
                continue
            out.append((width, pts * k))
    return out


def render_vectors(vectors: list[tuple[float, np.ndarray]], widths: set[float], shape: tuple[int, int],
                   thickness: int = 2) -> np.ndarray:
    """Ink image (255 = ink) of only the vector lines drawn with the given pen widths."""
    img = np.zeros(shape, np.uint8)
    lines = [np.round(pts).astype(np.int32) for w, pts in vectors if w in widths]
    if lines:
        cv2.polylines(img, lines, False, 255, thickness)
    return img


def _load_image(data: bytes) -> tuple[np.ndarray, float | None]:
    img = Image.open(io.BytesIO(data))
    dpi = None
    info_dpi = img.info.get("dpi")
    if info_dpi:
        d = float(info_dpi[0])
        # 72 and 96 are software defaults, not real scan resolutions.
        if 100 <= d <= 2400 and round(d) not in (72, 96):
            dpi = d
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    gray = np.asarray(img.convert("L"))
    return gray, dpi


def merge_number_words(words: list[Word]) -> list[Word]:
    """Join "13" + "400" into "13 400" when they sit next to each other on one line."""
    out: list[Word] = []
    for w in sorted(words, key=lambda w: (w.vertical, round(w.center[1] if not w.vertical else w.center[0]), w.box[0])):
        if out and not w.vertical and not out[-1].vertical and w.text.isdigit() and out[-1].text.replace(" ", "").isdigit():
            p = out[-1]
            h = p.box[3] - p.box[1]
            same_line = abs(w.center[1] - p.center[1]) < 0.4 * h
            close = 0 <= w.box[0] - p.box[2] < 0.8 * h
            if same_line and close and len(w.text) == 3:
                out[-1] = Word(f"{p.text} {w.text}", (p.box[0], min(p.box[1], w.box[1]), w.box[2], max(p.box[3], w.box[3])),
                               min(p.conf, w.conf))
                continue
        out.append(w)
    return out


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven paper tone and lighting so one threshold fits the whole page."""
    k = max(15, (min(gray.shape) // 60) | 1)
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255
    return np.clip(norm, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Otsu splits dark text from everything else, so pale grey linework on scans gets lost.
    # Anything clearly darker than the paper (beyond its noise) is ink.
    paper = float(np.median(blur))
    noise = 1.4826 * float(np.median(np.abs(blur[blur > paper - 20] - paper)))
    t = max(min(otsu, 200), paper - max(30.0, 5 * noise))
    return np.where(blur < t, 255, 0).astype(np.uint8)


def estimate_skew(gray: np.ndarray, max_deg: float = 6.0) -> float:
    """Dominant small rotation of long straight lines (degrees, counter-clockwise positive)."""
    # LSD works on grey levels; on a binarised scan it only finds short jagged pieces.
    scale = min(1.0, 2000 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(small)[0]
    if lines is None or len(lines) == 0:
        return 0.0
    l = lines.reshape(-1, 4)
    dx, dy = l[:, 2] - l[:, 0], l[:, 3] - l[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    ang = (ang + 45) % 90 - 45  # fold to [-45, 45): horizontal and vertical lines agree
    keep = (length > 0.02 * max(small.shape)) & (np.abs(ang) < max_deg)
    if keep.sum() < 5:
        return 0.0
    a, w = ang[keep], length[keep]
    order = np.argsort(a)
    cum = np.cumsum(w[order])
    # Weighted median. A line drawn rotated counter-clockwise has a negative angle here (image y
    # points down), and cv2's positive angle rotates counter-clockwise, so the median is the fix.
    return float(a[order][np.searchsorted(cum, cum[-1] / 2)])


def rotate(gray: np.ndarray, words: list[Word], deg: float) -> tuple[np.ndarray, list[Word]]:
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)
    moved = []
    for wd in words:
        cx, cy = wd.center
        nx, ny = M @ np.array([cx, cy, 1.0])
        hw, hh = (wd.box[2] - wd.box[0]) / 2, (wd.box[3] - wd.box[1]) / 2
        moved.append(Word(wd.text, (nx - hw, ny - hh, nx + hw, ny + hh), wd.conf, wd.vertical))
    return out, moved


def pen_width(ink: np.ndarray) -> float:
    """Most common thin-stroke width in px (from horizontal and vertical ink run lengths)."""
    runs = []
    for arr in (ink, ink.T):
        b = (arr > 0).astype(np.int8)
        d = np.diff(np.pad(b, ((0, 0), (1, 1))), axis=1)
        starts = np.argwhere(d == 1)
        ends = np.argwhere(d == -1)
        r = ends[:, 1] - starts[:, 1]
        runs.append(r[(r > 0) & (r < 30)])
    r = np.concatenate(runs)
    if len(r) == 0:
        return 2.0
    counts = np.bincount(r)
    return float(max(1, int(np.argmax(counts[1:])) + 1))
