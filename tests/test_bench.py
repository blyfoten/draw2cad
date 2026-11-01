"""The benchmark must score the truth as perfect and penalise each kind of error it claims to measure."""

from __future__ import annotations

import random

import cv2
import ezdxf
import numpy as np
import pytest

from bench.dxfio import read_dxf
from draw2cad.dxf import new_doc, plan_to_dxf
from draw2cad.plan import Plan
from bench.score import score_case, score_footprints
from bench.synth import generate_plan, make_case
from draw2cad.layers import Category, category_of_layer


@pytest.fixture(scope="module")
def plan() -> Plan:
    return generate_plan(random.Random(42))


def transformed(src, dst, sx=1.0, sy=1.0, dx=0.0, dy=0.0):
    """Copy a DXF with every entity scaled and moved (simulates a converter's scale/origin error)."""
    doc = ezdxf.readfile(src)
    m = ezdxf.math.Matrix44.chain(ezdxf.math.Matrix44.scale(sx, sy, 1), ezdxf.math.Matrix44.translate(dx, dy, 0))
    msp = doc.modelspace()
    for e in list(msp):
        if e.dxftype() == "DIMENSION" or (e.dxftype() == "ARC" and sx != sy):
            msp.delete_entity(e)  # CAD arcs cannot be scaled non-uniformly
        else:
            e.transform(m)
    doc.saveas(dst)


def test_truth_scores_perfect(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "p.dxf")
    s = score_case(read_dxf(tmp_path / "p.dxf"), plan)
    assert s.f1(25) > 0.99
    assert max(abs(v) for v in s.footprint_err_mm) < 1
    assert max(abs(v) for v in s.scale_err_pct) < 0.01
    assert s.doors.recall == 1 and s.doors.precision == 1
    assert s.windows.recall == 1 and s.windows.precision == 1


def test_origin_is_free(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "p.dxf")
    transformed(tmp_path / "p.dxf", tmp_path / "moved.dxf", dx=123456, dy=-7890)
    s = score_case(read_dxf(tmp_path / "moved.dxf"), plan)
    assert s.f1(25) > 0.99
    assert s.translation_mm == pytest.approx((-123456, 7890), abs=1)


def test_scale_error_is_measured(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "p.dxf")
    transformed(tmp_path / "p.dxf", tmp_path / "big.dxf", sx=1.02, sy=0.99)
    s = score_case(read_dxf(tmp_path / "big.dxf"), plan)
    assert s.scale_err_pct[0] == pytest.approx(2.0, abs=0.05)
    assert s.scale_err_pct[1] == pytest.approx(-1.0, abs=0.05)
    W = plan.meta["outer_width"]
    assert s.footprint_err_mm[0] == pytest.approx(0.02 * W, abs=5)
    assert s.f1(25) < 0.9


def test_large_scale_error_is_measured(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "p.dxf")
    transformed(tmp_path / "p.dxf", tmp_path / "half.dxf", sx=0.5, sy=0.5)
    s = score_case(read_dxf(tmp_path / "half.dxf"), plan)
    assert s.scale_err_pct == pytest.approx((-50.0, -50.0), abs=0.5)


def test_missing_wall_lowers_completeness(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "full.dxf")
    interior = [w for w in plan.walls if not w.exterior]
    assert interior, "fixture plan should have interior walls"
    gone = interior[0].id
    partial = Plan(walls=[w for w in plan.walls if w.id != gone],
                   openings=[o for o in plan.openings if o.wall != gone], rooms=[], dims=[])
    plan_to_dxf(partial, tmp_path / "partial.dxf")
    full = score_case(read_dxf(tmp_path / "full.dxf"), plan)
    part = score_case(read_dxf(tmp_path / "partial.dxf"), plan)
    assert part.completeness[50] < full.completeness[50] - 0.02
    assert part.precision[50] > 0.99  # what is drawn is still right
    assert part.doors.recall < 1


def test_wrong_layers_lower_ext_int_accuracy(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "p.dxf")
    doc = ezdxf.readfile(tmp_path / "p.dxf")
    for e in doc.modelspace():
        if e.dxf.layer in ("A-VAGG-YTTER", "A-VAGG-INNER"):
            e.dxf.layer = "A-VAGG"
    doc.saveas(tmp_path / "generic.dxf")
    s = score_case(read_dxf(tmp_path / "generic.dxf"), plan)
    assert s.f1(25) > 0.99
    assert s.ext_int_accuracy == 0


def test_metre_units_are_converted(plan, tmp_path):
    plan_to_dxf(plan, tmp_path / "p.dxf")
    transformed(tmp_path / "p.dxf", tmp_path / "m.dxf", sx=0.001, sy=0.001)
    doc = ezdxf.readfile(tmp_path / "m.dxf")
    doc.header["$INSUNITS"] = 6
    doc.saveas(tmp_path / "m.dxf")
    assert score_case(read_dxf(tmp_path / "m.dxf"), plan).f1(25) > 0.99


def test_footprint_check(tmp_path):
    doc = new_doc()
    msp = doc.modelspace()
    for x0 in (0, 30000):  # two plans on one sheet, the second one 12 x 9 m
        w, d = (13400, 8000) if x0 == 0 else (12000, 9000)
        msp.add_lwpolyline([(x0, 0), (x0 + w, 0), (x0 + w, d), (x0, d)], close=True,
                           dxfattribs={"layer": "A-VAGG-YTTER"})
    doc.saveas(tmp_path / "fp.dxf")
    [res] = score_footprints(read_dxf(tmp_path / "fp.dxf"), [{"name": "a", "width": 13400, "depth": 8000}])
    assert res["found"] and abs(res["width_err_mm"]) < 1 and abs(res["depth_err_mm"]) < 1


@pytest.mark.parametrize("level", ["clean", "scan", "rough"])
def test_case_generation(tmp_path, level):
    make_case(tmp_path / level, seed=7, level=level)
    gt = Plan.from_json(tmp_path / level / "gt.json")
    T = np.array(gt.meta["world_to_px"])
    assert (tmp_path / level / "image.png").exists() and (tmp_path / level / "gt.dxf").exists()
    # The stored world->pixel map must put the building inside the image.
    x0, y0, x1, y1 = gt.bounds()
    corners = np.array([[x0, y0, 1], [x1, y1, 1]]).T
    px = (T @ corners) if T.shape == (2, 3) else (T @ corners)[:2] / (T @ corners)[2]
    h, w = cv2.imread(str(tmp_path / level / "image.png"), cv2.IMREAD_GRAYSCALE).shape
    assert (px[0] > 0).all() and (px[0] < w).all() and (px[1] > 0).all() and (px[1] < h).all()


def test_layer_categories():
    assert category_of_layer("A-VAGG-YTTER") is Category.WALL_EXT
    assert category_of_layer("WALLS_EXTERIOR") is Category.WALL_EXT
    assert category_of_layer("A-WALL-INTR") is Category.WALL_INT
    assert category_of_layer("DOORS") is Category.DOOR
    assert category_of_layer("A-FÖNSTER") is Category.WINDOW
    assert category_of_layer("0") is Category.OTHER
