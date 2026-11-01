"""Local OCR (RapidOCR / ONNX, no network) for room names, dimensions and scale notes."""

from __future__ import annotations

import re
from functools import lru_cache

import cv2
import numpy as np

from draw2cad.raster import Word, merge_number_words


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _run(img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
    res, _ = _engine()(img)
    return [(np.asarray(box, dtype=float), text, float(conf)) for box, text, conf in (res or [])]


def read_text(gray: np.ndarray, min_conf: float = 0.5) -> list[Word]:
    """Horizontal text from the page, plus vertical (bottom-to-top) text from a rotated pass."""
    words: list[Word] = []
    for box, text, conf in _run(gray):
        if conf < min_conf:
            continue
        x0, y0 = box.min(0)
        x1, y1 = box.max(0)
        if (y1 - y0) > 1.5 * (x1 - x0) and len(text) > 1:
            continue  # vertical text read sideways; the rotated pass reads it properly
        words.append(Word(text.strip(), (x0, y0, x1, y1), conf))

    h = gray.shape[0]
    rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)  # bottom-to-top text becomes horizontal
    for box, text, conf in _run(rotated):
        if conf < min_conf:
            continue
        # A point (xr, yr) in the rotated image is (yr, h - 1 - xr) in the page.
        pts = np.column_stack([box[:, 1], h - 1 - box[:, 0]])
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        if (y1 - y0) < 1.2 * (x1 - x0) and len(text) > 1:
            continue  # horizontal text, already read by the first pass
        words.append(Word(text.strip(), (x0, y0, x1, y1), conf, vertical=True))
    return merge_number_words(words)


_NUM = re.compile(r"^\d{1,3}(?:[ . ]?\d{3})*$|^\d+$")


def parse_mm(text: str) -> int | None:
    """'13 400' / '13400' / '13.400' -> 13400; anything with letters -> None."""
    t = text.strip().replace(" ", " ")
    if not _NUM.match(t):
        return None
    v = int(re.sub(r"[ . ]", "", t))
    return v if 100 <= v <= 200_000 else None


_SCALE = re.compile(r"1\s*:\s*(\d{1,4})")
_PAPER = re.compile(r"\bA\s?([0-4])\b")


def find_scale_notes(words: list[Word]) -> list[int]:
    """Drawing scales written on the sheet; notes next to the word SKALA count double."""
    out = []
    for w in words:
        for m in _SCALE.finditer(w.text):
            v = int(m.group(1))
            if v in (20, 25, 50, 100, 200, 250, 400, 500, 1000):  # 1:1 etc. is never a plan scale
                out += [v, v] if "SKALA" in w.text.upper() or "SCALE" in w.text.upper() else [v]
    return out


def find_paper_notes(words: list[Word]) -> list[str]:
    out = []
    for w in words:
        if _SCALE.search(w.text) or "SKALA" in w.text.upper():
            out += [f"A{m.group(1)}" for m in _PAPER.finditer(w.text.upper())]
    return out
