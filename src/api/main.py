"""
src/api/main.py
===============
TOmni-Care Agent — FastAPI 服務入口

端點：
  GET  /health          → 健康檢查
  POST /api/chat        → 文字輸入，同步回覆（Gradio / REST 客戶端用）
  WS   /ws/chat         → WebSocket 音訊串流（音訊 chunk → ASR+情緒+LLM+TTS）

WebSocket 協議（JSON 訊息格式）：
  Client → Server：
    {"type": "audio_chunk", "data": "<base64 PCM bytes>"}
    {"type": "text", "content": "腹肚痛，是按怎？"}
    {"type": "reset"}  # 清除對話歷史

  Server → Client：
    {"type": "transcript",  "text": "..."}
    {"type": "emotion",     "label": "anxious", "confidence": 0.82}
    {"type": "response",    "text": "...", "token": "..."}  # streaming token
    {"type": "audio",       "data": "<base64 WAV bytes>"}
    {"type": "error",       "message": "..."}
    {"type": "done"}
"""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

# ── Lifespan：啟動時預載 Agent ────────────────────────────────
_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    global _agent
    from .agent_singleton import get_agent
    _agent = await get_agent()
    print("[API] TOmniCareAgent 初始化完成 ✓")
    yield
    # shutdown（目前無需清理）

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="TOmni-Care Agent API",
    description="台語長者照護 AI 助理 — FastAPI 服務",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生產環境應限制來源
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST 端點 ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    emotion_label: str
    emotion_confidence: float
    rag_sources: list = []


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TOmni-Care Agent", "version": "0.4.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    同步文字對話端點。

    適用場景：Gradio UI、curl 測試、一般 REST 客戶端。
    不支援音訊串流，音訊請用 WebSocket /ws/chat。
    """
    global _agent
    if _agent is None:
        from .agent_singleton import get_agent
        _agent = await get_agent()

    result = await _agent.run_text(req.text)
    return ChatResponse(
        response=result.get("response", "歹勢，系統暫時無法回覆。"),
        emotion_label=result.get("emotion_label", "calm"),
        emotion_confidence=result.get("emotion_confidence", 1.0),
        rag_sources=result.get("rag_sources", []),
    )


# ── WebSocket 端點 ────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """
    WebSocket 即時語音串流端點。

    完整流程：
      音訊 chunk (base64) → VAD 累積 → ASR → 情緒偵測 → RAG → LLM 串流 → TTS → 音訊回傳

    連線管理：
      每個連線維護獨立的 TOmniCareAgent 實例（獨立歷史記憶）。
    """
    from .websocket_handler import WebSocketSession

    await ws.accept()
    session = WebSocketSession(ws)
    print(f"[WS] 新連線建立")

    try:
        await session.run()
    except WebSocketDisconnect:
        print(f"[WS] 連線已斷開")
    except Exception as e:
        print(f"[WS] 未預期錯誤：{e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
