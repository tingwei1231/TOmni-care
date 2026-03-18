"""
src/agent/state.py — LangGraph AgentState 定義

設計：TypedDict(total=False) 讓各節點只更新自己負責的欄位。

Graph 路由邏輯：
  emotion_node → needs_comfort=True  → COMFORT 模式（安撫優先）
  emotion_node → needs_comfort=False → CHAT 模式（RAG 知識問答）
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class AgentMode(str, Enum):
    CHAT = "chat"
    COMFORT = "comfort"


class AgentState(TypedDict, total=False):
    # 輸入
    audio_path: Optional[str]
    audio_array: Optional[Any]
    text_input: Optional[str]
    # 中間結果
    transcript: Optional[str]
    emotion_label: Optional[str]       # calm / anxious / sad / angry
    emotion_confidence: float
    needs_comfort: bool
    rag_context: Optional[str]
    rag_sources: List[str]
    # 控制
    mode: AgentMode
    should_use_rag: bool
    # 輸出
    response: Optional[str]
    audio_response: Optional[bytes]
    # 系統
    history: List[Dict[str, str]]
    turn_count: int
    error: Optional[str]


def initial_state(audio_path=None, text_input=None) -> AgentState:
    return AgentState(
        audio_path=audio_path, audio_array=None, text_input=text_input,
        transcript=None, emotion_label="calm", emotion_confidence=1.0,
        needs_comfort=False, rag_context=None, rag_sources=[],
        mode=AgentMode.CHAT, should_use_rag=True,
        response=None, audio_response=None,
        history=[], turn_count=0, error=None,
    )
