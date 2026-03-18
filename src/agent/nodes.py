"""
src/agent/nodes.py — LangGraph 節點實作

節點圖（簡化）：
  START → asr_node → emotion_node ─[條件]→ rag_node → llm_node → (tts_node) → END

各節點只更新自己負責的欄位，不修改其他欄位。
"""
from __future__ import annotations
from .state import AgentMode, AgentState


async def asr_node(state: AgentState) -> AgentState:
    """
    語音→文字節點。
    text_input 有值 → 直接 passthrough（跳過 ASR）。
    """
    if state.get("text_input"):
        return {**state, "transcript": state["text_input"]}
    audio_path = state.get("audio_path")
    audio_array = state.get("audio_array")
    if not audio_path and audio_array is None:
        return {**state, "error": "無音訊輸入（audio_path 與 audio_array 皆為空）"}
    try:
        from ..asr.transcriber import TaiwaneseASR
        asr = TaiwaneseASR(model_size="medium", device="auto")
        if audio_path:
            result = await asr.transcribe(audio_path)
        else:
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            sf.write(tmp, audio_array, 16000)
            result = await asr.transcribe(tmp)
        return {**state, "transcript": result.full_text}
    except Exception as e:
        return {**state, "error": f"ASR 失敗：{e}"}


async def emotion_node(state: AgentState) -> AgentState:
    """
    情緒感知節點。
    音訊輸入 → EmotionDetector（SVM or 規則型 fallback）
    文字輸入 → 台語情感詞典 fallback
    """
    audio_path = state.get("audio_path")
    audio_array = state.get("audio_array")
    if audio_path or audio_array is not None:
        try:
            from ..emotion.detector import EmotionDetector
            detector = EmotionDetector()
            pred = await detector.detect(audio_path or audio_array,
                                         sr=None if audio_path else 16000)
            mode = AgentMode.COMFORT if pred.needs_comfort else AgentMode.CHAT
            return {**state, "emotion_label": pred.label.value,
                    "emotion_confidence": pred.confidence,
                    "needs_comfort": pred.needs_comfort, "mode": mode}
        except Exception as e:
            print(f"[Agent] 情緒偵測失敗（{e}），預設 CALM")
    # 文字情感詞典 fallback
    text = state.get("transcript", "")
    anxious_kw = ["真痛","真驚","按怎辦","揣無","袂記得","真歹勢"]
    sad_kw = ["寂寞","真艱苦","毋知按怎","哭","想無"]
    if any(k in text for k in anxious_kw):
        label, nc, mode = "anxious", True, AgentMode.COMFORT
    elif any(k in text for k in sad_kw):
        label, nc, mode = "sad", True, AgentMode.COMFORT
    else:
        label, nc, mode = "calm", False, AgentMode.CHAT
    return {**state, "emotion_label": label, "emotion_confidence": 0.6,
            "needs_comfort": nc, "mode": mode}


async def rag_node(state: AgentState) -> AgentState:
    """
    RAG 知識庫工具呼叫節點。
    關鍵詞觸發機制：只有醫療/生活關鍵詞才執行向量搜尋，避免閒聊問題注入無關知識。
    """
    text = state.get("transcript", "")
    trigger_kw = ["痛","藥","病","醫","血壓","血糖","頭殼","腹肚","心臟","跋倒","睏","食","飲食","運動"]
    if not any(k in text for k in trigger_kw):
        return {**state, "should_use_rag": False, "rag_context": None, "rag_sources": []}
    try:
        from ..rag.retriever import ChromaRetriever
        retriever = ChromaRetriever()
        chunks = await retriever.retrieve(text)
        return {**state, "should_use_rag": True,
                "rag_context": retriever.format_context(chunks),
                "rag_sources": [c.source for c in chunks]}
    except Exception as e:
        print(f"[Agent] RAG 失敗（{e}）")
        return {**state, "should_use_rag": False, "rag_context": None, "rag_sources": []}


async def llm_node(state: AgentState) -> AgentState:
    """
    台語回覆生成節點。
    mode=CHAT    → 標準 PromptTemplate（RAG + Few-shot）
    mode=COMFORT → 安撫 Prompt（情感優先，資訊次之）
    """
    text = state.get("transcript", "")
    rag_ctx = state.get("rag_context")
    mode = state.get("mode", AgentMode.CHAT)
    history_raw = state.get("history", [])
    try:
        from ..llm.client import LLMClient, LLMConfig, Message
        from ..llm.prompts import PromptTemplate, build_comfort_prompt
        client = LLMClient(LLMConfig())
        hist = [Message(role=m["role"], content=m["content"]) for m in history_raw[-12:]]
        if mode == AgentMode.COMFORT:
            messages = build_comfort_prompt(text, rag_ctx)
            messages = messages[:-1] + hist + [messages[-1]]
        else:
            messages = PromptTemplate().build(text, rag_ctx, hist)
        response = await client.chat(messages)
        new_hist = (history_raw + [{"role":"user","content":text},
                                   {"role":"assistant","content":response}])[-40:]
        return {**state, "response": response, "history": new_hist,
                "turn_count": state.get("turn_count", 0) + 1}
    except Exception as e:
        return {**state, "response": f"歹勢，系統發生問題：{e}", "error": str(e)}


async def tts_node(state: AgentState) -> AgentState:
    """TTS 語音合成節點（可選，第四階段 WebSocket 整合後使用）。"""
    resp = state.get("response", "")
    if not resp:
        return state
    try:
        from ..tts.synthesizer import TaiwanTTS
        tts = TaiwanTTS()
        audio = await tts.synthesize_to_bytes(resp)
        return {**state, "audio_response": audio}
    except Exception as e:
        print(f"[Agent] TTS 失敗（{e}），跳過")
        return state
