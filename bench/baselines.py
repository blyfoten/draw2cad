"""Reference converters for calibrating the benchmark. A converter is `f(image: Path, out_dxf: Path) -> None`.

- oracle: writes the ground truth itself (synthetic cases only). Must score perfectly.
- naive:  threshold + thick-stroke contours at an assumed 1:100 / 300 dpi. A floor to beat.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from draw2cad.dxf import new_doc, plan_to_dxf
from draw2cad.plan import Plan
from draw2cad.layers import DEFAULT_LAYERS, Category


def oracle(image: Path, out_dxf: Path) -> None:
    gt = image.parent / "gt.json"
    if not gt.exists():
        raise FileNotFoundError("oracle only works on synthetic cases (no gt.json next to the image)")
    plan_to_dxf(Plan.from_json(gt), out_dxf)


def naive(image: Path, out_dxf: Path, scale: int = 100, dpi: int = 300) -> None:
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thick = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(thick, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mm_per_px = 25.4 / dpi * scale
    h = gray.shape[0]
    doc = new_doc()
    msp = doc.modelspace()
    for c in contours:
        if cv2.contourArea(c) < 30:
            continue
        poly = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
        pts = [(x * mm_per_px, (h - y) * mm_per_px) for x, y in poly]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": DEFAULT_LAYERS[Category.WALL]})
    doc.saveas(out_dxf)
