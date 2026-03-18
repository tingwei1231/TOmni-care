"""
src/agent/graph.py — LangGraph 狀態機建構

節點架構：
  START → asr_node → emotion_node ──[條件]──→ rag_node → llm_node → (tts_node) → END
                                          ↘ error → END

條件路由：
  route_after_emotion: error→END, else→rag_node
  route_after_rag:     error→END, else→llm_node
  route_after_llm:     有音訊→tts_node, else→END（enable_tts=True 時生效）

TOmniCareAgent：高階封裝，支援 run_audio()/run_text() 雙模式 + 多輪歷史。
"""
from __future__ import annotations
from typing import Literal
from langgraph.graph import END, START, StateGraph
from .state import AgentState
from .nodes import asr_node, emotion_node, rag_node, llm_node, tts_node


def route_after_emotion(state: AgentState) -> Literal["rag_node", "__end__"]:
    return "__end__" if state.get("error") else "rag_node"

def route_after_rag(state: AgentState) -> Literal["llm_node", "__end__"]:
    return "__end__" if state.get("error") else "llm_node"

def route_after_llm(state: AgentState) -> Literal["tts_node", "__end__"]:
    if (state.get("audio_path") or state.get("audio_array") is not None) and state.get("response"):
        return "tts_node"
    return "__end__"


def build_graph(enable_tts: bool = False):
    """
    建構並編譯 TOmni-Care Agent 狀態機。

    Parameters
    ----------
    enable_tts : bool
        是否啟用 TTS 節點（預設關閉，第四階段 WebSocket 整合後開啟）
    """
    g = StateGraph(AgentState)
    g.add_node("asr_node", asr_node)
    g.add_node("emotion_node", emotion_node)
    g.add_node("rag_node", rag_node)
    g.add_node("llm_node", llm_node)
    if enable_tts:
        g.add_node("tts_node", tts_node)

    g.add_edge(START, "asr_node")
    g.add_edge("asr_node", "emotion_node")
    g.add_conditional_edges("emotion_node", route_after_emotion,
                            {"rag_node": "rag_node", "__end__": END})
    g.add_conditional_edges("rag_node", route_after_rag,
                            {"llm_node": "llm_node", "__end__": END})
    if enable_tts:
        g.add_conditional_edges("llm_node", route_after_llm,
                                {"tts_node": "tts_node", "__end__": END})
        g.add_edge("tts_node", END)
    else:
        g.add_edge("llm_node", END)

    return g.compile()


class TOmniCareAgent:
    """
    TOmni-Care Agent 高階封裝。
    自動維護多輪對話歷史（最近 20 輪）。

    Usage
    -----
    agent = TOmniCareAgent()
    result = await agent.run_text("腹肚痛，是按怎？")
    print(result["response"])
    print(result["emotion_label"])   # calm / anxious / sad / angry
    """
    def __init__(self, enable_tts=False):
        self._graph = build_graph(enable_tts=enable_tts)
        self._history: list = []

    async def run_audio(self, audio_path: str) -> AgentState:
        from .state import initial_state
        state = initial_state(audio_path=audio_path)
        state["history"] = self._history
        result = await self._graph.ainvoke(state)
        self._history = result.get("history", [])
        return result

    async def run_text(self, text: str) -> AgentState:
        from .state import initial_state
        state = initial_state(text_input=text)
        state["history"] = self._history
        result = await self._graph.ainvoke(state)
        self._history = result.get("history", [])
        return result

    def reset(self):
        self._history = []


# 快速驗證：python src/agent/graph.py
async def _demo():
    import asyncio
    agent = TOmniCareAgent()
    tests = [
        ("平靜查詢", "藥仔食完了，愛按怎？"),
        ("焦慮案例", "我腹肚真痛，真驚，按怎辦？"),
        ("閒聊", "你好，今仔日天氣真好。"),
    ]
    for name, text in tests:
        print(f"\n{'='*40}\n📣 [{name}] {text}")
        r = await agent.run_text(text)
        print(f"🧠 情緒：{r.get('emotion_label')} | 模式：{r.get('mode')}")
        print(f"🤖 {(r.get('response') or '')[:80]}...")

if __name__ == "__main__":
    import asyncio
    asyncio.run(_demo())
