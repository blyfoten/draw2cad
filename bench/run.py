"""Run a converter over the benchmark and print scores.

    python -m bench.run --converter bench.baselines:naive
    python -m bench.run --converter bench.baselines:oracle --no-real

Writes every output DXF plus results.json under runs/<converter name>/.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
import traceback
from pathlib import Path

import yaml

from bench.dxfio import read_dxf
from draw2cad.plan import Plan
from bench.score import score_case, score_footprints

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}


def load_converter(spec: str):
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func)


def fmt(v, spec: str = "6.1f") -> str:
    return "     -" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def run_synthetic(convert, cases: Path, out: Path) -> list[dict]:
    rows = []
    for case in sorted(p for p in cases.iterdir() if (p / "gt.json").exists()):
        dxf = out / f"{case.name}.dxf"
        row = {"case": case.name}
        t0 = time.perf_counter()
        try:
            convert(case / "image.png", dxf)
            row["seconds"] = time.perf_counter() - t0
            row.update(score_case(read_dxf(dxf), Plan.from_json(case / "gt.json")).to_dict())
        except Exception as e:  # a crashing converter is a result, not a benchmark failure
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
        rows.append(row)
    return rows


def run_real(convert, real: Path, out: Path) -> list[dict]:
    checks = yaml.safe_load((real / "checks.yaml").read_text(encoding="utf-8")) or {}
    rows = []
    for img in sorted(p for p in real.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        dxf = out / f"real_{img.stem}.dxf"
        row = {"case": img.name}
        t0 = time.perf_counter()
        try:
            convert(img, dxf)
            row["seconds"] = time.perf_counter() - t0
            lw = read_dxf(dxf)
            row["wall_polylines"] = sum(len(v) for k, v in lw.lines.items() if k.value.startswith("wall"))
            row["footprints"] = score_footprints(lw, (checks.get(img.name) or {}).get("footprints", []))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def print_synthetic(rows: list[dict]) -> None:
    print("\nSynthetic cases (mm unless noted; F1/completeness/precision at 50 mm)")
    head = f"{'case':22} {'dW':>7} {'dD':>7} {'scX%':>6} {'scY%':>6} {'med':>6} {'p95':>6} " \
           f"{'compl':>6} {'prec':>6} {'F1':>6} {'doorR':>6} {'winR':>6} {'ext/int':>7}"
    print(head)
    print("-" * len(head))
    f1s = []
    for r in rows:
        if "error" in r:
            print(f"{r['case']:22} ERROR {r['error']}")
            f1s.append(0.0)
            continue
        f1s.append(r["f1_50"])
        dw, dd = r["footprint_err_mm"]
        sx, sy = r["scale_err_pct"]
        print(f"{r['case']:22} {fmt(dw, '7.0f')} {fmt(dd, '7.0f')} {fmt(sx, '6.2f')} {fmt(sy, '6.2f')} "
              f"{fmt(r['face_err_median_mm'])} {fmt(r['face_err_p95_mm'])} "
              f"{r['completeness']['50']:6.2f} {r['precision']['50']:6.2f} {r['f1_50']:6.2f} "
              f"{r['doors']['recall']:6.2f} {r['windows']['recall']:6.2f} {r['ext_int_accuracy']:7.2f}")
    if f1s:
        print(f"\nmean F1@50mm: {sum(f1s) / len(f1s):.3f} over {len(f1s)} cases")


def print_real(rows: list[dict]) -> None:
    print("\nReal samples (known dimensions only)")
    for r in rows:
        if "error" in r:
            print(f"  {r['case']}: ERROR {r['error']}")
            continue
        fps = r["footprints"]
        if not fps:
            print(f"  {r['case']}: ran, {r['wall_polylines']} wall polylines, no checks")
        for fp in fps:
            if fp["found"]:
                print(f"  {r['case']}: {fp['name']}: width {fp['width_err_mm']:+.0f} mm, "
                      f"depth {fp['depth_err_mm']:+.0f} mm")
            else:
                print(f"  {r['case']}: {fp['name']}: no walls found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converter", required=True, help="module:function, e.g. bench.baselines:naive")
    ap.add_argument("--cases", type=Path, default=Path("bench/cases"))
    ap.add_argument("--real", type=Path, default=Path("data/real"))
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    convert = load_converter(args.converter)
    out = args.out or Path("runs") / args.converter.replace(":", "_").replace(".", "_")
    out.mkdir(parents=True, exist_ok=True)

    synthetic = run_synthetic(convert, args.cases, out) if args.cases.exists() else []
    real = [] if args.no_real else run_real(convert, args.real, out)
    # JSON keys must be strings; normalise before printing so both views agree.
    result = json.loads(json.dumps({"converter": args.converter, "synthetic": synthetic, "real": real}))
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if synthetic:
        print_synthetic(result["synthetic"])
    if real:
        print_real(result["real"])
    print(f"\nOutputs and results.json in {out}")


if __name__ == "__main__":
    main()
