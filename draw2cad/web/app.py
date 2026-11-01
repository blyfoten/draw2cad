"""Local web app: upload a drawing, inspect the reconstruction layer by layer, download DXF.

    python -m draw2cad.web            # http://127.0.0.1:8765
"""

from __future__ import annotations

import io
import math
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from shapely.geometry import Polygon
from shapely.ops import unary_union

from draw2cad import agent_cc, ai
from draw2cad.agent import Session, run_agent
from draw2cad.convert import Options, Prepared, Result, analyze_prepared, prepare
from draw2cad.dxf import door_symbol, iter_polygons, plan_to_dxf, window_symbol
from draw2cad.feedback import FeedbackLog
from draw2cad.layers import DEFAULT_LAYERS, Category

STATIC = Path(__file__).parent / "static"
MAX_JOBS = 20

app = FastAPI(title="draw2cad")


@dataclass
class Job:
    filename: str
    prep: Prepared
    opts: Options
    result: Result
    agent: dict | None = None  # {"running", "events", "summary", "quality", "cost_usd"}


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()


def _store(job: Job) -> str:
    jid = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[jid] = job
        while len(JOBS) > MAX_JOBS:
            JOBS.pop(next(iter(JOBS)))
    return jid


def _job(jid: str) -> Job:
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "Unknown or expired job; upload the drawing again.")
    return job


def _r(v: float) -> float:
    return round(float(v), 1)


def overlay(res: Result) -> dict:
    """The reconstructed plan in page pixels, per layer, for drawing over the scan."""
    plan = res.plan
    px = lambda p: [_r(c) for c in res.to_px(*p)]
    layers: dict[str, list] = {c.value: [] for c in Category}

    cuts = unary_union([o.cut(plan.wall(o.wall)) for o in plan.openings]) if plan.openings else None
    ext = unary_union([w.polygon() for w in plan.walls if w.exterior])
    inner = unary_union([w.polygon() for w in plan.walls if not w.exterior]).difference(ext)
    for solid, cat in ((ext, Category.WALL_EXT), (inner, Category.WALL_INT)):
        if cuts is not None:
            solid = solid.difference(cuts)
        for poly in iter_polygons(solid):
            rings = [[px(c) for c in poly.exterior.coords]] + [[px(c) for c in h.coords] for h in poly.interiors]
            layers[cat.value].append({"type": "poly", "rings": rings})

    for o in plan.openings:
        wall = plan.wall(o.wall)
        if o.kind == "door":
            hinge, leaf, closed = door_symbol(wall, o)
            r = o.width / res.scale.mean
            layers[Category.DOOR.value].append({"type": "door", "hinge": px(hinge), "leaf": px(leaf),
                                                "closed": px(closed), "r": _r(r), "width": o.width})
        elif o.kind == "window":
            layers[Category.WINDOW.value].append({"type": "lines", "segs": [[px(p), px(q)] for p, q in window_symbol(wall, o)],
                                                  "width": o.width})

    for room in plan.rooms:
        poly = Polygon(room.polygon)
        c = poly.representative_point()
        layers[Category.ROOM.value].append({"type": "room", "ring": [px(p) for p in room.polygon], "name": room.name,
                                            "area": round(poly.area / 1e6, 1), "at": px((c.x, c.y))})
    for t in plan.texts:
        layers[Category.TEXT.value].append({"type": "text", "text": t.text, "at": px(t.pos),
                                            "size": _r(t.height / res.scale.mean), "angle": t.angle})
    layers[Category.OTHER.value] = [{"type": "lines", "segs": [[px(p) for p in pl] for pl in plan.lines]}] if plan.lines else []
    for chain in plan.dims:
        p, q = chain.points[0], chain.points[-1]
        off = chain.offset
        layers[Category.DIMENSION.value].append({
            "type": "dim", "p": px((p[0] + off[0], p[1] + off[1])), "q": px((q[0] + off[0], q[1] + off[1])),
            "e1": [px(p), px((p[0] + off[0], p[1] + off[1]))], "e2": [px(q), px((q[0] + off[0], q[1] + off[1]))],
            "label": f"{round(math.dist(p, q)):,}".replace(",", " ")})
    return layers


def summary(jid: str, job: Job) -> dict:
    res = job.result
    plan = res.plan
    H, W = res.page.gray.shape
    counts = {
        Category.WALL_EXT.value: sum(w.exterior for w in plan.walls),
        Category.WALL_INT.value: sum(not w.exterior for w in plan.walls),
        Category.DOOR.value: sum(o.kind == "door" for o in plan.openings),
        Category.WINDOW.value: sum(o.kind == "window" for o in plan.openings),
        Category.ROOM.value: len(plan.rooms),
        Category.TEXT.value: len(plan.texts),
        Category.DIMENSION.value: len(plan.dims),
        Category.OTHER.value: len(plan.lines),
    }
    return {
        "id": jid,
        "filename": job.filename,
        "width": W,
        "height": H,
        "pages": res.page.pages,
        "page": job.opts.page,
        "layers": [{"key": c.value, "name": DEFAULT_LAYERS[c], "count": counts.get(c.value, 0)}
                   for c in (Category.WALL_EXT, Category.WALL_INT, Category.DOOR, Category.WINDOW, Category.ROOM,
                             Category.DIMENSION, Category.TEXT, Category.OTHER)],
        "scale": {"sx": res.scale.sx, "sy": res.scale.sy, "source": res.scale.source,
                  "confidence": res.scale.confidence, "notes": res.scale.notes},
        "style": res.walls.style,
        "skew": round(res.page.skew_deg, 2),
        "ai": {"enabled": job.opts.use_ai, "ok": res.ai is not None, "error": res.ai_error,
               "drawing_type": (res.ai or {}).get("drawing_type"), "notes": (res.ai or {}).get("notes")},
        "rooms": [{"name": r.name, "area": round(Polygon(r.polygon).area / 1e6, 1)} for r in plan.rooms],
        "warnings": res.warnings,
        "timings": {k: round(v, 1) for k, v in res.timings.items()},
        "overlay": overlay(res),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"ai_available": ai.available(), "claude_code": agent_cc.status()}


@app.post("/api/convert")
async def convert(file: UploadFile = File(...), use_ai: bool = Form(True), page: int = Form(0)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    opts = Options(use_ai=use_ai, page=page)
    try:
        prep = await run_in_threadpool(prepare, data, file.filename or "upload", opts)
        res = await run_in_threadpool(analyze_prepared, prep, opts)
    except Exception as e:
        raise HTTPException(422, f"Could not read this file as a drawing: {e}")
    job = Job(file.filename or "drawing", prep, opts, res)
    return summary(_store(job), job)


class Calibration(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]
    mm: float


@app.post("/api/jobs/{jid}/calibrate")
async def calibrate(jid: str, cal: Calibration):
    job = _job(jid)
    d = math.dist(cal.p1, cal.p2)
    if d < 5 or cal.mm <= 0:
        raise HTTPException(400, "Pick two points further apart and enter a positive distance.")
    opts = Options(use_ai=job.opts.use_ai, page=job.opts.page, user_scale={"mm_per_px": cal.mm / d},
                   params=job.opts.params)
    job.result = await run_in_threadpool(analyze_prepared, job.prep, opts)
    job.opts = opts
    return summary(jid, job)


@app.post("/api/jobs/{jid}/reset-scale")
async def reset_scale(jid: str):
    job = _job(jid)
    opts = Options(use_ai=job.opts.use_ai, page=job.opts.page, params=job.opts.params)
    job.result = await run_in_threadpool(analyze_prepared, job.prep, opts)
    job.opts = opts
    return summary(jid, job)


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return summary(jid, _job(jid))


def _agent_thread(jid: str, job: Job, backend: str) -> None:
    state = job.agent
    session = Session(job.prep, params=deepcopy(job.opts.params), user_scale=job.opts.user_scale)
    if backend == "api":  # the Claude Code backend's MCP server keeps its own feedback log
        session.feedback = FeedbackLog(job.filename, "api")

    def on_event(e: dict) -> None:
        if e["type"] == "tool":  # keep the log readable: tool name plus its key arguments
            e = {"type": "tool", "name": e["name"], "input": {k: v for k, v in e["input"].items() if k != "actions"},
                 "actions": len(e["input"].get("actions", []))}
        state["events"].append(e)
        if session.result is not None and e["type"] in ("result", "turn"):
            job.result = session.result  # the page shows each new reconstruction as it arrives
            state["version"] += 1
        if e["type"] == "done":
            state.update(summary=e["summary"], quality=e["quality"], cost_usd=e["cost_usd"])

    try:
        if backend == "claude-code":
            agent_cc.run_agent_claude_code(job.prep, on_event, session=session, drawing=job.filename)
        else:
            run_agent(job.prep, on_event, session=session)
        job.result = session.result
        job.opts = Options(use_ai=job.opts.use_ai, page=job.opts.page, user_scale=session.user_scale,
                           params=session.params)
    except Exception as e:
        state["events"].append({"type": "error", "text": f"{type(e).__name__}: {e}"})
    finally:
        fb = session.feedback.dir if session.feedback else getattr(session, "feedback_dir", None)
        state["feedback_dir"] = str(fb) if fb else None
        state["running"] = False
        state["version"] += 1


@app.post("/api/jobs/{jid}/agent")
def start_agent(jid: str, backend: str = "api"):
    """backend: "api" (Anthropic API, billed per token) or "claude-code" (the local claude CLI and
    its subscription login)."""
    job = _job(jid)
    if backend == "claude-code":
        st = agent_cc.status()
        if not st["available"]:
            raise HTTPException(400, f"Claude Code cannot be used: {st['reason']}.")
    elif not ai.available():
        raise HTTPException(400, "No ANTHROPIC_API_KEY configured.")
    if job.agent and job.agent["running"]:
        raise HTTPException(409, "The agent is already working on this drawing.")
    job.agent = {"running": True, "events": [], "version": 0, "summary": None, "quality": None,
                 "cost_usd": None, "backend": backend}
    threading.Thread(target=_agent_thread, args=(jid, job, backend), daemon=True).start()
    return {"started": True}


@app.get("/api/jobs/{jid}/agent")
def agent_status(jid: str, since: int = 0):
    job = _job(jid)
    st = job.agent or {"running": False, "events": [], "version": 0}
    return {"running": st["running"], "events": st["events"][since:], "next": len(st["events"]),
            "version": st["version"], "summary": st.get("summary"), "quality": st.get("quality"),
            "cost_usd": st.get("cost_usd"), "backend": st.get("backend"), "feedback_dir": st.get("feedback_dir")}


@app.get("/api/jobs/{jid}/image.png")
def image(jid: str):
    ok, png = cv2.imencode(".png", _job(jid).result.page.gray)
    return Response(png.tobytes(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/jobs/{jid}/drawing.dxf")
def dxf(jid: str, layers: str | None = None):
    job = _job(jid)
    fd, tmp_name = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        plan = job.result.plan
        if layers is not None:  # export only the layers that are switched on
            on = set(layers.split(","))
            plan = deepcopy(plan)
            if "wall_ext" not in on and "wall_int" not in on:
                plan.walls, plan.openings = [], []
            else:
                plan.walls = [w for w in plan.walls if ("wall_ext" if w.exterior else "wall_int") in on]
                ids = {w.id for w in plan.walls}
                plan.openings = [o for o in plan.openings if o.wall in ids and (o.kind == "opening" or o.kind in on)]
            plan.rooms = plan.rooms if "room" in on else []
            plan.texts = plan.texts if "text" in on else []
            plan.lines = plan.lines if "other" in on else []
            plan.dims = plan.dims if "dimension" in on else []
        plan_to_dxf(plan, tmp)
        data = tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)
    name = Path(job.filename).stem + ".dxf"
    return StreamingResponse(io.BytesIO(data), media_type="application/dxf",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
