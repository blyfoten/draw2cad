"""Edit tools the agent relies on (regressions from its own bug reports)."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

from bench.synth import make_case
from draw2cad.agent import Session, apply_edit, recompute_rooms, run_tool
from draw2cad.convert import Options, prepare


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    d = tmp_path_factory.mktemp("case")
    make_case(d, seed=1003, level="clean")
    data = (d / "image.png").read_bytes()
    s = Session(prepare(data, "image.png", Options(use_ai=False)))
    s.run()
    return s


def _longest_ext(res):
    return max((w for w in res.plan.walls if w.exterior), key=lambda w: w.length)


def test_ids_are_never_reused(session):
    res = session.run()
    plan = res.plan
    victim = plan.walls[0].id
    apply_edit(res, {"op": "remove_walls", "ids": [victim]})
    for _ in range(3):
        apply_edit(res, {"op": "add_wall", "x0": 100, "y0": 100, "x1": 300, "y1": 100, "thickness_mm": 95})
    ids = [w.id for w in plan.walls]
    assert len(ids) == len(set(ids))


def test_remove_region_cuts_only_inside(session):
    res = session.run()
    wall = _longest_ext(res)
    a, b = res.to_px(*wall.a), res.to_px(*wall.b)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    before = wall.length
    apply_edit(res, {"op": "remove_walls", "region": [mx - 20, my - 20, mx + 20, my + 20]})
    line = LineString([wall.a, wall.b])
    from shapely.geometry import Point

    rest = [w for w in res.plan.walls if w.exterior and line.distance(Point(w.a)) < 1 and line.distance(Point(w.b)) < 1]
    assert len(rest) == 2  # the parts on both sides of the cut stay
    assert before * 0.8 < sum(w.length for w in rest) < before


def test_opening_lands_on_the_wall_it_names(session):
    res = session.run()
    wall = _longest_ext(res)
    x, y = res.to_px(*wall.point_at(wall.length / 2))
    msg = apply_edit(res, {"op": "add_opening", "kind": "window", "x": x, "y": y, "width_mm": 1000})
    assert f"on wall {wall.id}" in msg
    o = res.plan.openings[-1]
    cx, cy = res.to_px(*o.center(res.plan.wall(o.wall)))
    assert abs(cx - x) < 2 and abs(cy - y) < 2


def test_opening_far_from_walls_is_rejected(session):
    res = session.run()
    assert "no wall" in apply_edit(res, {"op": "add_opening", "kind": "door", "x": 5, "y": 5, "width_mm": 900})


def test_room_names_survive_recompute(session):
    res = session.run()
    room = max(res.plan.rooms, key=lambda r: __import__("shapely").geometry.Polygon(r.polygon).area)
    from shapely.geometry import Polygon

    c = Polygon(room.polygon).representative_point()
    x, y = res.to_px(c.x, c.y)
    apply_edit(res, {"op": "rename_room", "x": x, "y": y, "name": "Testrum"})
    recompute_rooms(res)
    assert any(r.name == "TESTRUM" for r in res.plan.rooms)


def test_list_items_shows_ids(session):
    session.run()
    out = run_tool(session, "list_items", {"kind": "walls"})[0]["text"]
    assert "exterior" in out and out.split(":")[0].split()[0].isdigit()
