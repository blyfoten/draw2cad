"""What the agent learns about the tools while it works, saved for improving them.

Each agent run gets a folder under feedback/ with:
- issues.jsonl: one record per event: the agent's own issue reports (report_issue), every manual
  fix it made (each one is evidence of something the pipeline got wrong), each conversion run
  (parameters and flagged problems), and the final summary with the agent's tool feedback;
- crops: the drawing and the reconstruction around every reported issue and fix (before/after).

    python -m draw2cad.feedback            # collect all runs into feedback/REPORT.md

feedback/ holds pieces of real drawings, so it is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "feedback"

CATEGORIES = {
    "scale": "Scale",
    "walls": "Wall detection",
    "exterior": "Exterior / interior classification",
    "openings": "Doors and windows",
    "rooms": "Rooms and names",
    "text": "Text recognition (OCR)",
    "vector_pdf": "Vector PDF handling",
    "tools": "Agent tools and views (usability)",
    "other": "Other",
}


class FeedbackLog:
    def __init__(self, drawing: str, backend: str, root: Path = ROOT, folder: Path | None = None):
        stem = re.sub(r"[^\w.-]+", "_", Path(drawing).stem)[:60]
        self.dir = folder or root / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = len(list(self.dir.glob("*.png")))
        if folder is None:  # an existing folder (another process continuing the run) is already started
            self.write({"event": "start", "drawing": drawing, "backend": backend,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    def write(self, record: dict) -> None:
        with open(self.dir / "issues.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def image(self, png: bytes, label: str) -> str:
        self.n += 1
        name = f"{self.n:03d}_{label}.png"
        (self.dir / name).write_bytes(png)
        return name


def pad(region, margin: float = 120) -> list[float]:
    x0, y0, x1, y1 = region
    return [min(x0, x1) - margin, min(y0, y1) - margin, max(x0, x1) + margin, max(y0, y1) + margin]


def action_region(a: dict) -> list[float] | None:
    """Where an edit_plan action happened, in page px."""
    if a.get("region"):
        return pad(a["region"], 60)
    if all(k in a for k in ("x0", "y0", "x1", "y1")):
        return pad((a["x0"], a["y0"], a["x1"], a["y1"]))
    if "x" in a and "y" in a:
        return pad((a["x"], a["y"], a["x"], a["y"]), 200)
    return None


# --- report ----------------------------------------------------------------------------------------


def _load(run: Path) -> list[dict]:
    path = run / "issues.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


EDIT_MEANING = {
    "add_wall": ("walls", "missed wall (agent added it)"),
    "remove_walls": ("walls", "false wall (agent removed it)"),
    "add_opening": ("openings", "missed or misplaced opening (agent added it)"),
    "remove_openings": ("openings", "false or misplaced opening (agent removed it)"),
    "rename_room": ("rooms", "wrong or missing room name (agent renamed it)"),
}


def build_report(root: Path = ROOT) -> Path:
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    issues: dict[str, list] = defaultdict(list)
    edits: Counter = Counter()
    edit_examples: dict[str, list] = defaultdict(list)
    finishes = []
    for run in runs:
        records = _load(run)
        start = next((r for r in records if r["event"] == "start"), {})
        for r in records:
            where = f"{start.get('drawing', run.name)}"
            if r["event"] == "issue":
                issues[r.get("category", "other")].append((run, where, r))
            elif r["event"] == "edit":
                key = r["action"].get("op", "?")
                edits[key] += 1
                if len(edit_examples[key]) < 6:
                    edit_examples[key].append((run, where, r))
            elif r["event"] == "finish":
                finishes.append((run, where, r))

    lines = ["# draw2cad tool feedback", "",
             f"{len(runs)} agent runs, {sum(len(v) for v in issues.values())} reported issues, "
             f"{sum(edits.values())} manual fixes.", ""]
    lines += ["## Issues by tool area", "", "| Area | Reported issues | Major |", "|---|---|---|"]
    for cat, items in sorted(issues.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {CATEGORIES.get(cat, cat)} | {len(items)} | {sum(1 for *_, r in items if r.get('severity') == 'major')} |")
    lines += ["", "## Manual fixes (what the pipeline got wrong)", "", "| Fix | Count | Meaning |", "|---|---|---|"]
    for op, n in edits.most_common():
        lines.append(f"| {op} | {n} | {EDIT_MEANING.get(op, ('', op))[1]} |")

    for cat, items in sorted(issues.items(), key=lambda kv: -len(kv[1])):
        lines += ["", f"## {CATEGORIES.get(cat, cat)}", ""]
        for run, where, r in items:
            sev = r.get("severity", "")
            lines += [f"### {r.get('problem', '')[:100]}", "",
                      f"*{where}* · {sev} · run `{run.name}`", "",
                      f"- **Problem:** {r.get('problem', '')}",
                      f"- **Expected:** {r.get('expected', '')}",
                      f"- **Suggested improvement:** {r.get('suggestion', '')}", ""]
            imgs = [r[k] for k in ("input_png", "result_png") if r.get(k)]
            if imgs:
                lines.append(" ".join(f"![]({run.name}/{i})" for i in imgs))
                lines.append("")

    lines += ["", "## Examples of manual fixes", ""]
    for op, items in edit_examples.items():
        lines += [f"### {op}: {EDIT_MEANING.get(op, ('', op))[1]}", ""]
        for run, where, r in items:
            lines.append(f"*{where}*: `{json.dumps(r['action'], ensure_ascii=False)}` → {r.get('note', '')}")
            imgs = [r[k] for k in ("before_png", "after_png") if r.get(k)]
            if imgs:
                lines += ["", " ".join(f"![]({run.name}/{i})" for i in imgs)]
            lines.append("")

    lines += ["", "## The agent's feedback on the tools (per run)", ""]
    for run, where, r in finishes:
        lines += [f"### {where} ({r.get('quality', '')}, run `{run.name}`)", "",
                  r.get("tool_feedback") or "(none)", ""]

    out = root / "REPORT.md"
    root.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    print("wrote", build_report(args.root))


if __name__ == "__main__":
    main()
