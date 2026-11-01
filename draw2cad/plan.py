"""Ground-truth floor plan model. All coordinates are millimetres, Y pointing up (CAD convention)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

Pt = tuple[float, float]


@dataclass
class Wall:
    id: str
    a: Pt  # centerline start
    b: Pt  # centerline end
    thickness: float
    exterior: bool

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)

    @property
    def direction(self) -> Pt:
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        n = math.hypot(dx, dy)
        return dx / n, dy / n

    def point_at(self, offset: float) -> Pt:
        ux, uy = self.direction
        return self.a[0] + ux * offset, self.a[1] + uy * offset

    def polygon(self) -> Polygon:
        # Extended by half the thickness at both ends so corners and T-junctions fill solid.
        return LineString([self.a, self.b]).buffer(self.thickness / 2, cap_style="square", join_style="mitre")


@dataclass
class Opening:
    id: str
    kind: str  # "door" | "window" | "opening" (passage without a door)
    wall: str
    offset: float  # along the wall from `a` to the opening centre
    width: float  # clear opening in the wall
    swing: int = 1  # doors: +1 swings to the left of a->b, -1 to the right
    hinge_at_start: bool = True  # doors: hinge on the side nearer `a`

    def center(self, wall: Wall) -> Pt:
        return wall.point_at(self.offset)

    def cut(self, wall: Wall) -> Polygon:
        ux, uy = wall.direction
        c = self.center(wall)
        h = self.width / 2
        p, q = (c[0] - ux * h, c[1] - uy * h), (c[0] + ux * h, c[1] + uy * h)
        return LineString([p, q]).buffer(wall.thickness / 2 + 1, cap_style="flat")


@dataclass
class Room:
    name: str
    polygon: list[Pt]


@dataclass
class DimChain:
    """A dimension chain: points along a line, drawn offset from the building, with values in mm."""

    points: list[Pt]
    offset: Pt  # displacement vector of the dimension line from the measured points


@dataclass
class Text:
    text: str
    pos: Pt
    height: float
    angle: float = 0.0


@dataclass
class Plan:
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    dims: list[DimChain] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)  # annotations that are not room names
    lines: list[list[Pt]] = field(default_factory=list)  # other linework: furniture, fixtures, stairs
    meta: dict = field(default_factory=dict)

    def wall(self, wall_id: str) -> Wall:
        return next(w for w in self.walls if w.id == wall_id)

    def wall_solid(self):
        """Union of all walls minus door and window openings: what a correct DXF draws as wall."""
        solid = unary_union([w.polygon() for w in self.walls])
        cuts = [o.cut(self.wall(o.wall)) for o in self.openings]
        return solid.difference(unary_union(cuts)) if cuts else solid

    def bounds(self) -> tuple[float, float, float, float]:
        return unary_union([w.polygon() for w in self.walls]).bounds

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Plan:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            walls=[Wall(**{**w, "a": tuple(w["a"]), "b": tuple(w["b"])}) for w in d["walls"]],
            openings=[Opening(**o) for o in d["openings"]],
            rooms=[Room(r["name"], [tuple(p) for p in r["polygon"]]) for r in d["rooms"]],
            dims=[DimChain([tuple(p) for p in c["points"]], tuple(c["offset"])) for c in d["dims"]],
            texts=[Text(t["text"], tuple(t["pos"]), t["height"], t.get("angle", 0.0)) for t in d.get("texts", [])],
            lines=[[tuple(p) for p in pl] for pl in d.get("lines", [])],
            meta=d.get("meta", {}),
        )


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
