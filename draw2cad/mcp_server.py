"""draw2cad's agent tools as an MCP server, so Claude Code can be the agent.

    python -m draw2cad.mcp_server --prepared prep.pkl --state state_dir

`prep.pkl` is a pickled `convert.Prepared` (the drawing, already loaded and OCR'd). After every
tool call that changes the reconstruction, the session is pickled to `state_dir/state.pkl`, so
the web app can show progress and pick up the final result.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from draw2cad.agent import SYSTEM, TOOLS, Session, run_tool
from draw2cad.feedback import FeedbackLog

CHANGES = {"run_conversion", "edit_plan", "finish"}


def save_state(session: Session, state_dir: Path) -> None:
    tmp = state_dir / "state.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"result": session.result, "params": session.params, "user_scale": session.user_scale,
                     "runs": session.runs, "done": session.done}, f)
    os.replace(tmp, state_dir / "state.pkl")  # atomic: readers never see half a file


def _content(block: dict):
    if block["type"] == "image":
        return types.ImageContent(type="image", data=block["source"]["data"], mimeType=block["source"]["media_type"])
    return types.TextContent(type="text", text=block["text"])


def build_server(session: Session, state_dir: Path) -> Server:
    tools = [types.Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"]) for t in TOOLS]

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams):
        args = dict(params.arguments or {})
        try:
            # The pipeline is CPU-bound; keep the event loop responsive while it runs.
            blocks = await anyio.to_thread.run_sync(run_tool, session, params.name, args)
            is_error = False
        except Exception as e:
            blocks = [{"type": "text", "text": f"Tool failed: {type(e).__name__}: {e}"}]
            is_error = True
        if params.name in CHANGES and not is_error:
            save_state(session, state_dir)
        return types.CallToolResult(content=[_content(b) for b in blocks], isError=is_error)

    return Server("draw2cad", instructions=SYSTEM, on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--feedback", type=Path, help="folder for issue reports and fix crops (feedback.py)")
    ap.add_argument("--drawing", default="drawing", help="drawing name for the feedback log")
    args = ap.parse_args()
    with open(args.prepared, "rb") as f:
        session = Session(pickle.load(f))
    if args.feedback:
        session.feedback = FeedbackLog(args.drawing, "claude-code", folder=args.feedback)
    args.state.mkdir(parents=True, exist_ok=True)
    server = build_server(session, args.state)

    async def run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()
