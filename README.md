# draw2cad

Convert scanned, printed-to-PDF and hand-drawn floor plans into clean, correctly scaled, layered
DXF/DWG, using Swedish building standards (door modules, wall thicknesses) to correct what the
scan lost.

Status: **first working version**. Mean F1 0.52 at 50 mm on the synthetic benchmark (naive
baseline 0.09), with exact scale on 10 of 12 cases. On real sheets the outline and scale are good,
but thin interior walls are often missed, so rooms can merge.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use .venv/bin/python elsewhere
```

Optional: put `ANTHROPIC_API_KEY=...` in `.env` to let Claude read scale notes, overall dimensions
and room names. Everything else runs locally (RapidOCR, OpenCV).

## Use

```bash
python -m draw2cad.web                                   # web app on http://127.0.0.1:8765
python -m draw2cad.convert plan.pdf plan.dxf --no-ai     # command line
```

The web app shows the reconstruction over the scan layer by layer, lets you calibrate the scale
with two clicks and a known distance, and downloads DXF (all layers or only the visible ones).

### AI agent

```bash
python -m draw2cad.agent drawing.pdf out.dxf     # also: "Improve with AI agent" in the web app
```

Claude Opus 5.5 (`claude-opus-5-5`) drives the pipeline as tools (`draw2cad/agent.py`): it views
the drawing and the result drawn over it (with a pixel grid), reads the recognised text, lists the
pens of a vector PDF, re-runs the conversion with tuned parameters (`Params` in `convert.py`:
region, wall thickness range, which wall cues, which PDF pens, ...), and finally fixes local errors
by hand (add/remove walls and openings, rename rooms). Automatic checks guide it: room areas
against the areas written in the rooms, merged rooms, rooms without doors, room labels outside
every room. A run costs roughly $1–5 in API usage and takes a few minutes.

## How the converter works (`draw2cad/`)

1. `raster.py`: load PDF (PyMuPDF, page size known) or image; flatten paper tone, straighten, binarise.
2. `ocr.py`: local OCR, including a rotated pass for vertical dimension text.
3. `scale.py`: written dimensions measured against their dimension lines (chains are summed),
   checked against the scale note, sanity-checked against text size; refined later against the
   reconstructed outline.
4. `walls.py`: filled, outlined, hatched and grey-filled walls become axis-aligned wall bands;
   the thick walls define the building outline, which also decides exterior/interior.
5. `openings.py`: gaps become doors (swing arc beside it), windows (parallel glass lines) or openings.
6. `convert.py`: to millimetres, snapped to Swedish standards (`standards/se.yaml`), rooms from
   enclosed spaces with names from OCR (Å/Ä/Ö restored), text, other linework, overall dimensions.
7. `dxf.py`: layered DXF.

## Benchmark

```bash
python -m bench.synth --count 12 --seed 1                 # generate synthetic cases into bench/cases
python -m bench.run --converter draw2cad.convert:convert  # score the converter
python -m bench.run --converter bench.baselines:naive     # or a baseline
python -m pytest                                           # benchmark self-tests
```

A converter is any function `convert(image: Path, out_dxf: Path) -> None`, passed as `module:function`.

**Synthetic cases** (`bench/synth.py`) are random Swedish-style plans with exact ground truth:
exterior and interior walls in standard thicknesses, 9M/10M doors, module windows, room names
and dimension chains in the "13 400" style. They are rendered at 1:50 or 1:100 and 150-300 dpi,
with filled, outlined or hatched walls, then degraded: anisotropic scale, rotation, perspective,
low-resolution quantisation, blur, paper tone, noise and JPEG artefacts.

**Real samples** (`data/real/`) have no full ground truth. `data/real/checks.yaml` records the
dimensions printed on each drawing, and the benchmark checks the output against those.

**Scores** (`bench/score.py`), all after removing an arbitrary translation:

| Metric | Meaning |
| --- | --- |
| dW, dD | Outer width/depth error, mm |
| scX%, scY% | Scale error per axis, fitted separately (positive = drawn too large) |
| med, p95 | Distance from drawn wall faces to true faces, mm |
| compl / prec / F1 | Share of true wall faces found / drawn faces that are real, within 50 mm |
| doorR, winR | Doors and windows found (precision is in results.json) |
| ext/int | Wall faces on the right exterior/interior layer |

Reference points: the `oracle` baseline (writes the ground truth) scores F1 ≈ 1.0, and the `naive`
baseline (threshold + contours at an assumed 1:100 / 300 dpi) scores F1 ≈ 0.09.

## Layout

- `draw2cad/layers.py`: semantic categories and layer names (placeholders until SB-11 names are chosen)
- `draw2cad/standards/se.yaml`: Swedish dimension priors (**unverified starting values**)
- `bench/`: synthetic generator, DXF I/O, scorer, runner, baselines
- `data/real/`: real sample drawings and their known dimensions
