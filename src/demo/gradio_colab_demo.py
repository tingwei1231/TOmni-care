"""
src/demo/gradio_colab_demo.py
==============================
Colab 專用的 Gradio Demo — 不依賴本機 ASR/TTS 模型檔。

功能：
  • 文字輸入 → LangGraph Agent（RAG + 情緒偵測 + Groq LLM）→ 台語回覆
  • 上傳 WAV → Faster-Whisper ASR（若已下載模型）→ Agent → 回覆
  • 情緒狀態顯示（calm / anxious / sad / angry）
  • 多輪對話記憶（reset 按鈕清除）

在 Colab 執行：
  from src.demo.gradio_colab_demo import build_colab_ui
  demo = build_colab_ui()
  demo.launch(share=True)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parents[2]))

# ── 全域 Agent（只初始化一次）────────────────────────────────
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from src.agent.graph import TOmniCareAgent
        _agent = TOmniCareAgent(enable_tts=False)
    return _agent


def _run_async(coro):
    """在 Colab（已有 event loop）安全執行 async 函式。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Colab 環境：建新 thread 跑
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── 核心：文字輸入對話 ────────────────────────────────────────
def chat_text(user_input: str, history: list) -> tuple[str, list, str]:
    """
    Gradio ChatInterface 的 fn。
    Returns: (agent_response, updated_history, emotion_label)
    """
    if not user_input.strip():
        return "", history, "—"

    agent = _get_agent()
    result = _run_async(agent.run_text(user_input))

    response  = result.get("response", "歹勢，系統暫時無法回覆。")
    emotion   = result.get("emotion_label", "calm")
    confidence = result.get("emotion_confidence", 1.0)

    history.append((user_input, response))

    emotion_icons = {"calm": "😊 平靜", "anxious": "😰 焦慮",
                     "sad": "😢 悲傷", "angry": "😠 煩躁"}
    emotion_str = f"{emotion_icons.get(emotion, emotion)} ({confidence:.0%})"

    return "", history, emotion_str


# ── 核心：音訊輸入對話 ────────────────────────────────────────
def chat_audio(audio_path, history: list) -> tuple[list, str, str]:
    """
    上傳 WAV → ASR（若可用）→ Agent → 回覆
    """
    if audio_path is None:
        return history, "—", "請先上傳音訊。"

    agent = _get_agent()

    try:
        result = _run_async(agent.run_audio(audio_path))
    except Exception as e:
        return history, "—", f"❌ 錯誤：{e}"

    transcript = result.get("transcript", "（無法辨識）")
    response   = result.get("response",   "歹勢，系統暫時無法回覆。")
    emotion    = result.get("emotion_label", "calm")

    emotion_icons = {"calm": "😊 平靜", "anxious": "😰 焦慮",
                     "sad": "😢 悲傷", "angry": "😠 煩躁"}
    emotion_str = f"{emotion_icons.get(emotion, emotion)}"

    history.append((f"🎤 {transcript}", response))
    return history, emotion_str, f"📝 辨識：{transcript}"


def reset_agent_history():
    global _agent
    if _agent:
        _agent.reset()
    return [], "—", "✅ 對話記憶已清除"


# ── UI 建構 ───────────────────────────────────────────────────
def build_colab_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        font=[gr.themes.GoogleFont("Noto Sans TC"), "sans-serif"],
    )

    with gr.Blocks(theme=theme, title="TOmni-Care — 台語 AI 照護助理") as demo:
        gr.Markdown("""
        # 🏠 TOmni-Care — 台語 AI 照護助理
        > 台語輸入、情緒感知、在地醫療知識 | 技術棧：LangGraph + Groq + ChromaDB
        """)

        with gr.Row():
            emotion_display = gr.Textbox(
                label="🧠 情緒狀態", value="—",
                interactive=False, scale=1,
            )
            reset_btn = gr.Button("🔄 清除對話記憶", scale=1, variant="secondary")

        with gr.Tabs():

            # ── Tab 1: 文字對話（主要功能）──────────────────────
            with gr.Tab("💬 文字對話"):
                gr.Markdown("台語文字輸入 → LangGraph（情緒偵測 + RAG + LLM）→ 台語回覆")
                chatbot = gr.Chatbot(
                    label="TOmni-Care 對話",
                    height=420,
                    elem_id="chatbot",
                )
                with gr.Row():
                    text_inp = gr.Textbox(
                        placeholder="輸入台語（例：腹肚痛，是按怎？）",
                        label="台語輸入",
                        scale=4,
                        elem_id="text_input",
                    )
                    send_btn = gr.Button("送出 ▶", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        ["腹肚痛，是按怎？"],
                        ["藥仔食完了，愛按怎？"],
                        ["我真寂寞，無人𪜶。"],
                        ["血壓懸，有啥物食物愛注意？"],
                        ["今仔日天氣真好，你好無？"],
                    ],
                    inputs=[text_inp],
                    label="📝 範例問題",
                )

                send_btn.click(
                    fn=chat_text,
                    inputs=[text_inp, chatbot],
                    outputs=[text_inp, chatbot, emotion_display],
                )
                text_inp.submit(
                    fn=chat_text,
                    inputs=[text_inp, chatbot],
                    outputs=[text_inp, chatbot, emotion_display],
                )

            # ── Tab 2: 音訊上傳 ──────────────────────────────────
            with gr.Tab("🎤 音訊上傳"):
                gr.Markdown("""
                上傳 WAV 音訊 → ASR 辨識 → Agent 回覆

                > 格式：WAV、16kHz、Mono（手機錄音直接上傳即可）
                > ⚠ 需先下載 Whisper 模型（Cell 1-B）才能 ASR；若未下載，系統回報錯誤
                """)
                audio_chatbot = gr.Chatbot(label="音訊對話記錄", height=300)
                audio_inp = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="上傳或錄音",
                    elem_id="audio_input",
                )
                asr_status = gr.Textbox(label="辨識狀態", interactive=False)
                audio_btn = gr.Button("🎙 辨識並對話", variant="primary")

                audio_btn.click(
                    fn=chat_audio,
                    inputs=[audio_inp, audio_chatbot],
                    outputs=[audio_chatbot, emotion_display, asr_status],
                )

        reset_btn.click(
            fn=reset_agent_history,
            outputs=[chatbot, emotion_display, asr_status]
                    if 'asr_status' in dir() else [chatbot, emotion_display],
        )

        gr.Markdown("---\n**TOmni-Care Agent** v0.4 | LangGraph + Groq LLaMA-3.3-70B + ChromaDB")

    return demo


# ── 直接執行 ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_colab_ui()
    demo.launch(share=args.share, server_port=args.port, show_error=True)
