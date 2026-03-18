"""
src/api/websocket_handler.py
============================
WebSocket 連線管理：音訊 chunk 累積 → VAD → ASR → LangGraph → 串流回傳。

【串流架構】
  Client 傳送音訊 chunk（16kHz PCM base64）
    → BufferManager 累積
    → SileroVAD 偵測語音結束（靜音 500ms → 觸發推論）
    → asr_node（Faster-Whisper 辨識）
    → emotion_node（MFCC 情緒偵測）
    → rag_node（知識庫檢索）
    → llm_node（LLM 串流生成）→ 每個 token 即時回傳
    → tts_node（TTS 合成）→ WAV bytes 回傳

【訊息格式】
  Server → Client JSON：
    {"type":"transcript",  "text":"..."}
    {"type":"emotion",     "label":"anxious", "confidence":0.82, "needs_comfort":true}
    {"type":"response",    "token":"..."}    ← 串流 token
    {"type":"response_end","text":"..."}     ← 完整回覆（TTS 觸發後）
    {"type":"audio",       "data":"<base64 WAV>"}
    {"type":"done"}
    {"type":"error",       "message":"..."}
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

import numpy as np
from fastapi import WebSocket


class BufferManager:
    """音訊 chunk 累積器（16kHz mono float32）。"""

    SAMPLE_RATE = 16000
    # 最長累積時間（秒），防止無限累積
    MAX_DURATION_S = 30.0

    def __init__(self):
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0

    def append(self, pcm_bytes: bytes) -> None:
        """追加 PCM bytes（int16 little-endian）。"""
        arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer.append(arr)
        self._total_samples += len(arr)

    def get_audio(self) -> np.ndarray:
        """取得完整音訊陣列。"""
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._total_samples = 0

    @property
    def duration_s(self) -> float:
        return self._total_samples / self.SAMPLE_RATE

    @property
    def is_too_long(self) -> bool:
        return self.duration_s >= self.MAX_DURATION_S


class WebSocketSession:
    """
    管理單一 WebSocket 連線的完整生命週期。

    每個連線擁有獨立的 TOmniCareAgent（獨立對話歷史）。
    """

    SILENCE_THRESHOLD_S = 0.5   # 靜音超過 0.5 秒 → 觸發推論
    MIN_AUDIO_S = 1.0           # 最短有效語音（秒），太短跳過

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._buffer = BufferManager()
        self._last_speech_time = time.time()
        self._agent = None
        self._vad = None

    async def _get_agent(self):
        if self._agent is None:
            from ..agent.graph import TOmniCareAgent
            self._agent = TOmniCareAgent(enable_tts=False)
        return self._agent

    async def _send(self, data: dict) -> None:
        """安全傳送 JSON（連線已斷開時靜默忽略）。"""
        try:
            await self.ws.send_json(data)
        except Exception:
            pass

    async def _process_audio(self, audio: np.ndarray) -> None:
        """核心推論管線：音訊 → 回覆。"""
        agent = await self._get_agent()

        # ① ASR + Emotion（透過 LangGraph asr_node + emotion_node）
        # 直接使用 audio_array 輸入
        from ..agent.state import initial_state
        state = initial_state()
        state["audio_array"] = audio
        state["history"] = agent._history

        # 先跑 ASR
        from ..agent.nodes import asr_node, emotion_node
        state = await asr_node(state)
        transcript = state.get("transcript", "")
        if transcript:
            await self._send({"type": "transcript", "text": transcript})

        # 情緒偵測
        state = await emotion_node(state)
        await self._send({
            "type": "emotion",
            "label": state.get("emotion_label", "calm"),
            "confidence": state.get("emotion_confidence", 1.0),
            "needs_comfort": state.get("needs_comfort", False),
        })

        # ② RAG + LLM（用現有 RAGPipeline 串流，直接傳 transcript）
        if not transcript:
            return

        from ..llm.pipeline import RAGPipeline, LLMConfig
        from ..llm.client import LLMBackend
        pipeline = RAGPipeline(use_rag=True)

        comfort_mode = state.get("needs_comfort", False)
        full_tokens = []

        async for token in pipeline.stream_chat(transcript, comfort_mode=comfort_mode):
            full_tokens.append(token)
            await self._send({"type": "response", "token": token})

        full_response = "".join(full_tokens)
        agent._history = pipeline.history[-20:]  # 同步歷史

        await self._send({"type": "response_end", "text": full_response})
        await self._send({"type": "done"})

    async def run(self) -> None:
        """WebSocket 主迴圈。"""
        while True:
            raw = await self.ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send({"type": "error", "message": "無效 JSON"})
                continue

            msg_type = msg.get("type")

            # ── 音訊 chunk 累積 ───────────────────────────────
            if msg_type == "audio_chunk":
                pcm_bytes = base64.b64decode(msg.get("data", ""))
                self._buffer.append(pcm_bytes)
                self._last_speech_time = time.time()

                # 靜音超過門檻 或 緩衝太長 → 觸發推論
                if self._buffer.duration_s >= self.MIN_AUDIO_S and self._buffer.is_too_long:
                    audio = self._buffer.get_audio()
                    self._buffer.clear()
                    await self._process_audio(audio)

            # ── 靜音事件（Client 偵測到語音結束） ────────────
            elif msg_type == "vad_end":
                if self._buffer.duration_s >= self.MIN_AUDIO_S:
                    audio = self._buffer.get_audio()
                    self._buffer.clear()
                    await self._process_audio(audio)

            # ── 文字直接輸入 ──────────────────────────────────
            elif msg_type == "text":
                text = msg.get("content", "").strip()
                if not text:
                    continue
                agent = await self._get_agent()
                result = await agent.run_text(text)
                await self._send({"type": "transcript", "text": text})
                await self._send({
                    "type": "emotion",
                    "label": result.get("emotion_label", "calm"),
                    "confidence": result.get("emotion_confidence", 1.0),
                    "needs_comfort": result.get("needs_comfort", False),
                })
                await self._send({"type": "response_end", "text": result.get("response", "")})
                await self._send({"type": "done"})

            # ── 重置歷史 ──────────────────────────────────────
            elif msg_type == "reset":
                agent = await self._get_agent()
                agent.reset()
                self._buffer.clear()
                await self._send({"type": "done"})

            else:
                await self._send({"type": "error", "message": f"未知 type：{msg_type}"})
