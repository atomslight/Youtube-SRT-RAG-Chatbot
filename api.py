"""
api.py — FastAPI web server for YT RAG Chat.
Run: python api.py   (or: uvicorn api:app --host 0.0.0.0 --port 8000)
Then open: http://localhost:8000
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_core import RAGEngine, VideoSession

# ============================================================
# App lifecycle — engine & sessions
# ============================================================
engine: RAGEngine | None = None
sessions: dict[str, VideoSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading models...")
    engine = RAGEngine()
    print("Models ready. Server up at http://localhost:8000")
    yield
    sessions.clear()


app = FastAPI(title="YT RAG Chat", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# Schemas
# ============================================================
class ProcessRequest(BaseModel):
    url: str

class AskRequest(BaseModel):
    session_id: str
    query: str

# ============================================================
# SSE helpers
# ============================================================
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ============================================================
# Routes
# ============================================================
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/process")
async def process_video(body: ProcessRequest):
    session_id = str(uuid.uuid4())

    def stream():
        try:
            for event in engine.process_video(body.url):
                if event["type"] == "session":
                    s = event["session"]
                    sessions[session_id] = s
                    yield sse({
                        "type":        "ready",
                        "session_id":  session_id,
                        "video_title": s.video_title,
                        "video_id":    s.video_id,
                        "chunk_count": len(s.documents),
                    })
                else:
                    yield sse(event)
        except Exception as exc:
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/ask")
async def ask(body: AskRequest):
    session = sessions.get(body.session_id)

    if not session:
        def not_found():
            yield sse({"type": "error", "message": "Session not found. Please process a video first."})
        return StreamingResponse(not_found(), media_type="text/event-stream", headers=_SSE_HEADERS)

    def stream():
        try:
            for event in engine.answer_query(session, body.query):
                yield sse(event)
        except Exception as exc:
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"video_id": s.video_id, "video_title": s.video_title, "chunk_count": len(s.documents)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
