"""Optional drawing interpretation with Claude: what a person reads off the sheet at a glance.

Used only as a supplement: geometry always comes from measurement, and the AI's answers fill in
what measurement could not (room names OCR missed, scale when nothing is dimensioned).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

MODEL = "claude-opus-5-5"
MAX_SIDE = 1568  # px sent to the model; larger images are downscaled by the API anyway

SCHEMA = {
    "type": "object",
    "properties": {
        "drawing_type": {"type": "string", "description": "e.g. floor plan, section, elevation, site plan"},
        "scale_note": {"type": "string", "description": "Scale as written on the sheet, e.g. '1:100', or ''"},
        "paper_size": {"type": "string", "description": "Paper size if written (A1-A4), or ''"},
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "overall_width_mm": {"type": "number", "description": "Written overall horizontal dimension of the building's outer walls, 0 if not written"},
                    "overall_depth_mm": {"type": "number", "description": "Written overall vertical dimension of the building's outer walls, 0 if not written"},
                },
                "required": ["name", "overall_width_mm", "overall_depth_mm"],
                "additionalProperties": False,
            },
        },
        "rooms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Room name in Swedish as written, with correct å/ä/ö"},
                    "x": {"type": "number", "description": "Horizontal position of the room label, 0 = left edge, 1 = right edge"},
                    "y": {"type": "number", "description": "Vertical position of the room label, 0 = top edge, 1 = bottom edge"},
                },
                "required": ["name", "x", "y"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string", "description": "Anything that would affect a CAD conversion, in one or two sentences"},
    },
    "required": ["drawing_type", "scale_note", "paper_size", "plans", "rooms", "notes"],
    "additionalProperties": False,
}

PROMPT = """This is a scanned or printed architectural drawing, most likely a Swedish house plan.
Read it the way an architect would before redrawing it in CAD, and report:
- what kind of drawing it is;
- the drawing scale and paper size if they are written on the sheet (title block, "SKALA");
- for each floor plan on the sheet, the written overall outer dimensions (width = horizontal, depth = vertical), in mm;
- every room label inside the plans, with the position of the label as fractions of the image width and height;
- anything unusual that would affect a conversion.
Only report what is actually written or drawn; use 0 or '' when something is not there."""


def _load_env_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")


def available() -> bool:
    _load_env_key()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def analyze_drawing(gray: np.ndarray) -> dict:
    """Ask Claude to read the sheet. Raises on API errors; the caller decides how to degrade."""
    import anthropic

    _load_env_key()
    s = min(1.0, MAX_SIDE / max(gray.shape))
    img = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    ok, png = cv2.imencode(".png", img)
    data = base64.standard_b64encode(png.tobytes()).decode("ascii")

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to read this drawing")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
