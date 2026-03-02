"""
FastAPI server for the Scene Graph Planner demo.

Endpoints:
    GET  /                 -> Serve the frontend HTML
    POST /api/send         -> Send a user message; returns SSE stream
    POST /api/stop         -> Stop the current planning
    POST /api/reset        -> Reset environment  { env: "simple" | "vh" }
    GET  /api/graph        -> Current graph as PNG image
    GET  /api/graph/left   -> Navigate graph history left
    GET  /api/graph/right  -> Navigate graph history right
    GET  /api/random-task  -> Get a random task string

Run:
    uvicorn server:app --reload --port 7860
"""

import asyncio
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from demo_planner import PlannerApp

# ─────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────

app = FastAPI(title="Scene Graph Planner")
planner = PlannerApp()


# ─────────────────────────────────────────────────────────
#  Frontend
# ─────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("templates/index.html", media_type="text/html")


# ─────────────────────────────────────────────────────────
#  Chat / Planning
# ─────────────────────────────────────────────────────────

@app.post("/api/send")
async def send_message(request: Request):
    """
    Accept a user message and return an SSE stream of planner events.
    The planner generator runs in a background thread; events are
    forwarded to the async SSE stream via an asyncio.Queue.
    """
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "empty message"})

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _run():
        try:
            for event in planner.process_message(message):
                loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                q.put_nowait, {"type": "error", "text": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _generate():
        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/stop")
async def stop_planning():
    planner.stop()
    return JSONResponse({"status": "stopped"})


# ─────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────

@app.post("/api/reset")
async def reset_env(request: Request):
    data = await request.json()
    env = data.get("env", "simple")
    planner.reset_env(env)
    return JSONResponse({"status": "ok", "counter": planner.get_counter()})


# ─────────────────────────────────────────────────────────
#  Graph
# ─────────────────────────────────────────────────────────

@app.get("/api/graph")
async def get_graph():
    img_bytes = planner.graph_to_bytes()
    if not img_bytes:
        return Response(status_code=204)
    return Response(content=img_bytes, media_type="image/png")


@app.get("/api/graph/left")
async def graph_left():
    planner.step_left()
    return JSONResponse({"counter": planner.get_counter()})


@app.get("/api/graph/right")
async def graph_right():
    planner.step_right()
    return JSONResponse({"counter": planner.get_counter()})


# ─────────────────────────────────────────────────────────
#  Misc
# ─────────────────────────────────────────────────────────

@app.get("/api/random-task")
async def random_task():
    return JSONResponse({"task": planner.get_random_task()})


# ─────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=7860, reload=False)
