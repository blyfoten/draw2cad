"""Semantic categories for CAD entities and the layer names they are written to.

The default names are placeholders. Replace them with SB-11 (Svensk Byggtjänst CAD-lager,
based on ISO 13567) names, or your own office scheme, once that is decided. The benchmark
scores categories, not names, so renaming layers never changes scores.
"""

from __future__ import annotations

import re
from enum import Enum


class Category(str, Enum):
    WALL_EXT = "wall_ext"
    WALL_INT = "wall_int"
    WALL = "wall"  # wall of unknown kind (foreign files, naive converters)
    DOOR = "door"
    WINDOW = "window"
    DIMENSION = "dimension"
    TEXT = "text"
    ROOM = "room"
    OTHER = "other"


DEFAULT_LAYERS: dict[Category, str] = {
    Category.WALL_EXT: "A-VAGG-YTTER",
    Category.WALL_INT: "A-VAGG-INNER",
    Category.WALL: "A-VAGG",
    Category.DOOR: "A-DORR",
    Category.WINDOW: "A-FONSTER",
    Category.DIMENSION: "A-MATT",
    Category.TEXT: "A-TEXT",
    Category.ROOM: "A-RUM",
    Category.OTHER: "A-OVRIGT",
}

# Fallback patterns so the benchmark can also score DXF files from other tools
# (scan2cad, AIA-style A-WALL-EXTR, plain WALLS etc.). First match wins.
_PATTERNS: list[tuple[re.Pattern[str], Category]] = [
    (re.compile(r"(WALL|VAGG|VÄGG).*(EXT|YTTER)", re.I), Category.WALL_EXT),
    (re.compile(r"(WALL|VAGG|VÄGG).*(INT|INNER)", re.I), Category.WALL_INT),
    (re.compile(r"WALL|VAGG|VÄGG", re.I), Category.WALL),
    (re.compile(r"DOOR|DORR|DÖRR", re.I), Category.DOOR),
    (re.compile(r"WIND|GLAZ|FONSTER|FÖNSTER", re.I), Category.WINDOW),
    (re.compile(r"DIM|MATT|MÅTT", re.I), Category.DIMENSION),
    (re.compile(r"ROOM|RUM|AREA", re.I), Category.ROOM),
    (re.compile(r"TEXT|ANNO|LABEL", re.I), Category.TEXT),
    (re.compile(r"OVRIGT|ÖVRIGT|FURN|MÖBL|MOBL", re.I), Category.OTHER),
]

_BY_NAME = {name.upper(): cat for cat, name in DEFAULT_LAYERS.items()}


def category_of_layer(layer: str) -> Category:
    cat = _BY_NAME.get(layer.upper())
    if cat is not None:
        return cat
    for pattern, cat in _PATTERNS:
        if pattern.search(layer):
            return cat
    return Category.OTHER


WALL_CATEGORIES = {Category.WALL_EXT, Category.WALL_INT, Category.WALL}
