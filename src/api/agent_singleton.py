"""
src/api/agent_singleton.py
==========================
Agent 單例管理：確保全域只有一個 TOmniCareAgent 實例（避免重複載入模型）。
"""
from __future__ import annotations
import asyncio
from typing import Optional

_instance: Optional["TOmniCareAgent"] = None
_lock = asyncio.Lock()

async def get_agent():
    global _instance
    async with _lock:
        if _instance is None:
            from ..agent.graph import TOmniCareAgent
            _instance = TOmniCareAgent(enable_tts=False)
            print("[Singleton] TOmniCareAgent 建立完成")
    return _instance

def reset_agent():
    """測試用：重置單例。"""
    global _instance
    _instance = None
