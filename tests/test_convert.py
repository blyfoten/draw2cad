"""End-to-end regression guard: a clean synthetic plan must convert accurately (offline, no AI)."""

from __future__ import annotations

from bench.dxfio import read_dxf
from bench.score import score_case
from bench.synth import make_case
from draw2cad.convert import convert, room_spelling
from draw2cad.plan import Plan


def test_clean_plan_converts_to_scale(tmp_path):
    make_case(tmp_path / "case", seed=1003, level="clean")
    convert(tmp_path / "case" / "image.png", tmp_path / "out.dxf")
    s = score_case(read_dxf(tmp_path / "out.dxf"), Plan.from_json(tmp_path / "case" / "gt.json"))
    assert max(abs(v) for v in s.scale_err_pct) < 0.5
    assert max(abs(v) for v in s.footprint_err_mm) < 50
    assert s.f1(50) > 0.6
    assert s.ext_int_accuracy > 0.9


def test_room_spelling_restores_swedish_letters():
    assert room_spelling("TVATT") == "TVÄTT"
    assert room_spelling("GROVENTRE/TVATT") == "GROVENTRÉ/TVÄTT"
    assert room_spelling("SOV3") == "SOV 3"
    assert room_spelling("1:100") is None
