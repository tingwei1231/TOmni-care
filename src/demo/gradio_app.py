"""
src/demo/gradio_app.py
======================
TOmni-Care Agent 第一階段整合展示 — Gradio Web UI

運行方式：
  本機：python src/demo/gradio_app.py
  Colab：
    !git clone <repo>
    !pip install -r requirements.txt
    !python src/demo/gradio_app.py --share

功能：
  1. 麥克風錄音 → Faster-Whisper ASR → 台語文字
  2. 台語文字輸入 → Bert-VITS2 TTS（含變調）→ 播放音訊
  3. 完整 Round-trip 展示（語音輸入 → 語音輸出）

T4 GPU 優化注意事項：
  - ASR 使用 float16，約 2GB VRAM
  - TTS 使用 float16，約 2~3GB VRAM
  - 兩個模型合計在 T4 (16GB) 上可同時載入
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr
import soundfile as sf

# 確保 src/ 目錄在 import 路徑內
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.asr.transcriber import TaiwaneseASR
from src.asr.vad import SileroVAD
from src.tts.synthesizer import TaiwanTTS
from src.tts.tone_sandhi import process_phrase, parse_tl_syllable


# ── 全域模型實例（只載入一次，避免重複佔用 VRAM）────────────────
_asr: TaiwaneseASR | None = None
_tts: TaiwanTTS | None = None
_vad: SileroVAD | None = None


def get_models(
    asr_model_size: str = "medium",      # Colab T4 使用 medium 節省 VRAM
    tts_model_path: str | None = None,    # None → gTTS fallback
    device: str = "auto",
):
    """延遲初始化模型（第一次推論時才載入）。"""
    global _asr, _tts, _vad
    if _asr is None:
        _asr = TaiwaneseASR(
            model_size=asr_model_size,
            device=device,
            compute_type="float16" if device != "cpu" else "int8",
        )
    if _tts is None:
        _tts = TaiwanTTS(
            model_path=tts_model_path,
            device=device,
            speed=0.85,  # 長者專用，語速稍慢
        )
    if _vad is None:
        _vad = SileroVAD(device="cpu")  # VAD 永遠跑 CPU，節省 GPU


# ── ASR 推論函式（Gradio 的 fn 需為同步或支援 asyncio）────────────
def transcribe_audio(audio_path: str | None) -> str:
    """
    Gradio Audio 元件輸入 → 台語辨識文字輸出。

    Parameters
    ----------
    audio_path : str | None
        Gradio 傳入的暫存音檔路徑（WAV 格式）

    Returns
    -------
    str
        辨識結果文字
    """
    if audio_path is None:
        return "請先錄音或上傳音訊檔案。"

    get_models()

    try:
        # 使用 asyncio.run 執行非同步推論
        result = asyncio.run(_asr.transcribe(audio_path))

        if not result.full_text.strip():
            return "⚠ 未偵測到語音內容，請確認麥克風音量。"

        confidence_pct = (
            sum(s.confidence for s in result.segments) / len(result.segments) * 100
            if result.segments else 0
        )
        return (
            f"{result.full_text}\n\n"
            f"─── 診斷資訊 ───\n"
            f"語言偵測：{result.language} | "
            f"信心分數：{confidence_pct:.1f}% | "
            f"辨識耗時：{result.duration_ms:.0f}ms"
        )
    except Exception as e:
        return f"❌ 辨識失敗：{str(e)}"


# ── 變調展示函式 ───────────────────────────────────────────────
def demo_tone_sandhi(tl_phrase: str) -> str:
    """
    展示台語連讀變調。

    輸入格式（TLs，空白分隔音節）：
      tsia̍h8 pn̄g7     →   食飯
      tang1 png7 a5   →   當飯啊（範例）
    """
    if not tl_phrase.strip():
        return "請輸入 TLs 拼音（例：tang1 si7 bo5）"

    try:
        changed, original = process_phrase(tl_phrase)
        tokens = tl_phrase.strip().split()

        lines = ["📊 連讀變調結果：\n"]
        lines.append(f"{'音節':<12} {'原調':<6} {'變調後':<12}")
        lines.append("─" * 35)

        after_tokens = changed.split()
        for i, (orig, after) in enumerate(zip(tokens, after_tokens)):
            orig_syl = parse_tl_syllable(orig)
            after_syl = parse_tl_syllable(after)
            is_final = i == len(tokens) - 1
            marker = "（不變調）" if is_final else f"→ 調 {after_syl.tone}"
            lines.append(f"{orig:<12} 調 {orig_syl.tone:<4} {marker}")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ 處理失敗：{str(e)}"


# ── TTS 合成函式 ───────────────────────────────────────────────
def synthesize_speech(
    text: str,
    apply_sandhi: bool,
) -> tuple[str, tuple[int, np.ndarray] | None]:
    """
    台語文字 → 語音合成。

    Returns
    -------
    (status_msg, audio_tuple)
        audio_tuple : (sample_rate, audio_array) 供 Gradio Audio 播放
    """
    if not text.strip():
        return "請輸入台語文字。", None

    get_models()

    try:
        audio, sr = _tts.synthesize_sync(text, apply_tone_sandhi=apply_sandhi)
        duration = len(audio) / sr
        status = f"✅ 合成完成！音訊長度：{duration:.2f}s，取樣率：{sr}Hz"
        # Gradio Audio 接受 (sample_rate, np.ndarray) tuple
        return status, (sr, audio)
    except Exception as e:
        return f"❌ 合成失敗：{str(e)}", None


# ── Round-trip：語音輸入 → 語音輸出 ──────────────────────────────
def voice_to_voice(audio_path: str | None) -> tuple[str, tuple | None]:
    """
    完整 Round-trip 展示：
      麥克風錄音 → ASR → LLM（第二階段，此處用 echo 代替）→ TTS → 播放
    """
    if audio_path is None:
        return "請先錄音。", None

    # Step 1: ASR
    asr_result = transcribe_audio(audio_path)
    text = asr_result.split("\n\n")[0]  # 取第一行辨識結果

    if text.startswith("⚠") or text.startswith("❌"):
        return text, None

    # Step 2: 模擬 LLM 回覆（第二階段整合後替換）
    response_text = f"【AI 回覆（第二階段整合中）】\n你講：{text}"

    # Step 3: TTS
    status, audio = synthesize_speech(response_text, apply_sandhi=True)
    return f"辨識：{text}\n{status}", audio


# ── Gradio UI 定義 ─────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """建立 Gradio Blocks UI。"""

    # 台語主題配色（溫暖橙色，適合長者視覺）
    custom_theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        font=[gr.themes.GoogleFont("Noto Sans TC"), "sans-serif"],
    )

    with gr.Blocks(
        theme=custom_theme,
        title="TOmni-Care — 台語 AI 照護助理",
    ) as demo:

        gr.Markdown(
            """
            # 🏠 TOmni-Care — 台語 AI 照護助理
            ### 第一階段：語音核心展示（ASR + TTS）
            > 專為台灣長者設計的在地化 AI 語音助理
            """
        )

        with gr.Tabs():

            # ── Tab 1: ASR 台語辨識 ──────────────────────────
            with gr.Tab("🎤 台語辨識（ASR）"):
                gr.Markdown("### 錄音或上傳音訊 → 台語文字")
                with gr.Row():
                    with gr.Column(scale=1):
                        asr_audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="音訊輸入（WAV/MP3）",
                            elem_id="asr_audio_input",
                        )
                        asr_btn = gr.Button(
                            "🔍 開始辨識",
                            variant="primary",
                            elem_id="asr_btn",
                        )
                    with gr.Column(scale=1):
                        asr_output = gr.Textbox(
                            label="辨識結果",
                            lines=8,
                            placeholder="辨識文字將顯示於此...",
                            elem_id="asr_output",
                        )

                asr_btn.click(
                    fn=transcribe_audio,
                    inputs=[asr_audio_input],
                    outputs=[asr_output],
                )

            # ── Tab 2: 變調展示 ──────────────────────────────
            with gr.Tab("🎵 連讀變調（Tone Sandhi）"):
                gr.Markdown(
                    "### 台語連讀變調規則展示\n"
                    "輸入 TLs 拼音（以空白分隔音節），查看變調結果"
                )
                with gr.Row():
                    with gr.Column():
                        sandhi_input = gr.Textbox(
                            label="TLs 拼音輸入",
                            placeholder="例：tsia̍h8 pn̄g7",
                            elem_id="sandhi_input",
                        )
                        gr.Examples(
                            examples=[
                                ["tsia̍h8 pn̄g7"],
                                ["tang1 si7 bo5"],
                                ["li2 ho2 bo5"],
                                ["pak4 too2 thiann1"],
                            ],
                            inputs=[sandhi_input],
                            label="範例：",
                        )
                        sandhi_btn = gr.Button(
                            "⚡ 套用變調規則",
                            variant="primary",
                            elem_id="sandhi_btn",
                        )
                    with gr.Column():
                        sandhi_output = gr.Textbox(
                            label="變調結果",
                            lines=10,
                            elem_id="sandhi_output",
                        )

                sandhi_btn.click(
                    fn=demo_tone_sandhi,
                    inputs=[sandhi_input],
                    outputs=[sandhi_output],
                )

            # ── Tab 3: TTS 台語合成 ──────────────────────────
            with gr.Tab("🔊 台語合成（TTS）"):
                gr.Markdown("### 台語文字 → 語音合成")
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_input = gr.Textbox(
                            label="台語文字輸入",
                            placeholder="例：食飯真好，你好無？",
                            lines=3,
                            elem_id="tts_input",
                        )
                        apply_sandhi_checkbox = gr.Checkbox(
                            label="套用連讀變調",
                            value=True,
                            elem_id="apply_sandhi_checkbox",
                        )
                        tts_btn = gr.Button(
                            "🎙️ 開始合成",
                            variant="primary",
                            elem_id="tts_btn",
                        )
                    with gr.Column(scale=1):
                        tts_status = gr.Textbox(
                            label="合成狀態",
                            lines=2,
                            elem_id="tts_status",
                        )
                        tts_audio_output = gr.Audio(
                            label="合成語音",
                            type="numpy",
                            elem_id="tts_audio_output",
                        )

                tts_btn.click(
                    fn=synthesize_speech,
                    inputs=[tts_input, apply_sandhi_checkbox],
                    outputs=[tts_status, tts_audio_output],
                )

            # ── Tab 4: Round-trip 完整展示 ───────────────────
            with gr.Tab("🔄 完整 Round-Trip"):
                gr.Markdown(
                    "### 語音輸入 → 辨識 → AI 回覆 → 語音輸出\n"
                    "> ⚠ 第二階段整合前，AI 回覆以 Echo 方式呈現"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        rt_audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="麥克風輸入",
                            elem_id="rt_audio_input",
                        )
                        rt_btn = gr.Button(
                            "▶ 開始對話",
                            variant="primary",
                            size="lg",
                            elem_id="rt_btn",
                        )
                    with gr.Column(scale=1):
                        rt_status = gr.Textbox(
                            label="對話狀態",
                            lines=4,
                            elem_id="rt_status",
                        )
                        rt_audio_output = gr.Audio(
                            label="AI 語音回覆",
                            type="numpy",
                            elem_id="rt_audio_output",
                        )

                rt_btn.click(
                    fn=voice_to_voice,
                    inputs=[rt_audio_input],
                    outputs=[rt_status, rt_audio_output],
                )

        gr.Markdown(
            """
            ---
            **TOmni-Care Agent** | 第一階段：語音核心 | 技術棧：Faster-Whisper + Bert-VITS2 + Gradio
            """
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TOmni-Care Gradio Demo")
    parser.add_argument("--share", action="store_true", help="建立公開可訪問的 Gradio 連結（Colab 用）")
    parser.add_argument("--host", default="0.0.0.0", help="伺服器位址")
    parser.add_argument("--port", type=int, default=7860, help="埠號")
    parser.add_argument(
        "--asr-model",
        default="medium",
        help="Faster-Whisper 模型大小（tiny/base/small/medium/large-v3）",
    )
    parser.add_argument("--tts-model", default=None, help="Bert-VITS2 模型目錄（None=gTTS fallback）")
    args = parser.parse_args()

    # 設定環境變數供 get_models() 讀取
    os.environ["ASR_MODEL_SIZE"] = args.asr_model
    if args.tts_model:
        os.environ["TTS_MODEL_PATH"] = args.tts_model

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
