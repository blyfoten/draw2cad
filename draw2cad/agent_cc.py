"""Run the draw2cad agent through Claude Code (headless) instead of the Anthropic API.

When the `claude` CLI is installed and logged in with a Claude subscription on the machine that
runs draw2cad, Claude Code acts as the agent: it gets draw2cad's tools from `mcp_server.py` (and
no other tools), and its usage counts against the subscription's limits instead of API credits.

Only the unmodified `claude` binary is used, with the login that is already on the machine;
draw2cad never reads or stores the credentials. ANTHROPIC_API_KEY is removed from the child
process environment, because Claude Code would otherwise bill the API instead of the subscription.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from draw2cad.agent import MODEL, SYSTEM, TOOLS, Event, Session
from draw2cad.convert import Prepared
from draw2cad.feedback import FeedbackLog

TIME_LIMIT_S = 20 * 60


def claude_cli() -> str | None:
    return shutil.which("claude")


def status() -> dict:
    """Whether Claude Code can be used here, and as whom (no secrets are read)."""
    exe = claude_cli()
    if not exe:
        return {"available": False, "reason": "the claude CLI is not installed on this machine"}
    try:
        out = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, timeout=30, env=_child_env())
        info = json.loads(out.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"available": False, "reason": f"could not ask the claude CLI for its login status ({e})"}
    if not info.get("loggedIn"):
        return {"available": False, "reason": "the claude CLI is not logged in (run `claude` and use /login)"}
    if info.get("authMethod") != "claude.ai":
        return {"available": False, "reason": f"the claude CLI uses {info.get('authMethod')}, not a subscription login"}
    return {"available": True, "account": info.get("email"), "subscription": info.get("subscriptionType")}


def _child_env() -> dict:
    env = dict(os.environ)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):  # these would make Claude Code bill the API
        env.pop(k, None)
    return env


def run_agent_claude_code(prep: Prepared, on_event: Event | None = None, session: Session | None = None,
                          time_limit_s: int = TIME_LIMIT_S, drawing: str = "drawing") -> Session:
    """The agent's feedback (issue reports, fixes, crops) goes to a new folder under feedback/,
    written by the MCP server process."""
    exe = claude_cli()
    if not exe:
        raise RuntimeError("the claude CLI is not installed")
    session = session or Session(prep)
    emit = on_event or (lambda e: None)
    work = Path(tempfile.mkdtemp(prefix="draw2cad_cc_"))
    state_dir = work / "state"
    with open(work / "prep.pkl", "wb") as f:
        pickle.dump(prep, f)
    fb_dir = FeedbackLog(drawing, "claude-code").dir  # created here so the caller knows where it is
    session.feedback_dir = fb_dir  # type: ignore[attr-defined]
    config = {"mcpServers": {"draw2cad": {
        "command": sys.executable,
        "args": ["-m", "draw2cad.mcp_server", "--prepared", str(work / "prep.pkl"), "--state", str(state_dir),
                 "--feedback", str(fb_dir), "--drawing", drawing]}}}
    (work / "mcp.json").write_text(json.dumps(config), encoding="utf-8")

    page = prep.page
    prompt = (f"Convert the drawing loaded in the draw2cad tools to CAD. Page {page.gray.shape[1]} x "
              f"{page.gray.shape[0]} px{', vector PDF' if page.vectors else ''}"
              f"{f', sheet {page.paper_mm[0]:.0f} x {page.paper_mm[1]:.0f} mm' if page.paper_mm else ''}. "
              f"Use only the draw2cad tools and end by calling finish.")
    if prep.ai:
        prompt += f" A first quick reading of the sheet: {json.dumps(prep.ai, ensure_ascii=False)}"
    allowed = [f"mcp__draw2cad__{t['name']}" for t in TOOLS]
    cmd = [exe, "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--mcp-config", str(work / "mcp.json"), "--strict-mcp-config",
           "--tools", "", "--allowedTools", ",".join(allowed),
           "--model", MODEL, "--append-system-prompt", SYSTEM, "--no-session-persistence"]

    state_file = state_dir / "state.pkl"
    seen = {"mtime": 0.0}

    def load_state() -> bool:
        try:
            m = state_file.stat().st_mtime
        except FileNotFoundError:
            return False
        if m <= seen["mtime"]:
            return False
        try:
            with open(state_file, "rb") as f:
                st = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            return False  # caught mid-replace; next poll gets it
        seen["mtime"] = m
        session.result, session.params, session.user_scale = st["result"], st["params"], st["user_scale"]
        session.runs, session.done = st["runs"], st["done"]
        return True

    proc = subprocess.Popen(cmd, cwd=work, env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    stop = threading.Event()

    def watch_state():
        while not stop.wait(1.0):
            if load_state():
                emit({"type": "result", "runs": session.runs})

    watcher = threading.Thread(target=watch_state, daemon=True)
    watcher.start()
    timer = threading.Timer(time_limit_s, proc.kill)
    timer.start()
    final: dict = {}
    try:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "assistant":
                for b in msg.get("message", {}).get("content", []):
                    if b.get("type") == "text" and b.get("text", "").strip():
                        emit({"type": "say", "text": b["text"].strip()})
                    elif b.get("type") == "tool_use":
                        name = b.get("name", "").removeprefix("mcp__draw2cad__")
                        emit({"type": "tool", "name": name, "input": b.get("input", {})})
            elif msg.get("type") == "result":
                final = msg
        proc.wait()
    finally:
        timer.cancel()
        stop.set()
        watcher.join(timeout=5)
    load_state()
    err = proc.stderr.read() if proc.stderr else ""
    if final.get("is_error") or proc.returncode not in (0, None):
        reason = final.get("result") or err.strip()[-500:] or f"exit code {proc.returncode}"
        if proc.returncode == -9 or (proc.returncode and time_limit_s and not final):
            reason = reason or "time limit reached"
        emit({"type": "error", "text": f"Claude Code: {reason}"})
    if session.result is None:
        session.run()
    FeedbackLog(drawing, "claude-code", folder=fb_dir).write(
        {"event": "usage", "backend": "claude-code", "usage": final.get("usage", {}),
         "duration_s": round(final.get("duration_ms", 0) / 1000), "turns": final.get("num_turns")})
    emit({"type": "done", "summary": (session.done or {}).get("summary", final.get("result") or "Stopped."),
          "quality": (session.done or {}).get("quality", "unknown"), "usage": final.get("usage", {}),
          "cost_usd": None, "backend": "claude-code"})
    shutil.rmtree(work, ignore_errors=True)
    return session
