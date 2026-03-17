"""
src/asr/transcriber.py
======================
基於 Faster-Whisper 的台語語音辨識模組。

Faster-Whisper 使用 CTranslate2 引擎，相比原版 Whisper 在同等 GPU
記憶體下速度快 2~4 倍，非常適合在 T4 / L4 等 Colab 環境中使用。

台語辨識策略：
  - 使用 whisper-large-v3（對南島語系表現最佳）
  - language 固定設為 "zh"（中文），讓 Whisper 同時處理台羅混排文字
  - 若使用 Fine-tuned 台語模型，直接替換 model_size_or_path 即可
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


@dataclass
class TranscriptionSegment:
    """單一語音片段的辨識結果。"""

    start: float          # 片段起始時間（秒）
    end: float            # 片段結束時間（秒）
    text: str             # 辨識文字（台語漢字或台羅混排）
    confidence: float     # 平均 log-prob 信心分數（0~1 正規化）
    language: str = "zh"  # 偵測到的語言代碼


@dataclass
class TranscriptionResult:
    """完整語音辨識結果。"""

    segments: List[TranscriptionSegment]
    full_text: str
    language: str
    duration_ms: float   # 辨識耗時（毫秒），用於效能監控


class TaiwaneseASR:
    """
    台語語音辨識器。

    使用方式
    --------
    >>> asr = TaiwaneseASR(model_size="large-v3", device="cuda")
    >>> result = await asr.transcribe("path/to/audio.wav")
    >>> print(result.full_text)
    """

    # 台語辨識最佳化 beam search 參數
    _BEAM_SIZE = 5
    # initial_prompt：給 Whisper 注入台語語境，引導輸出台語漢字而非普通話
    _INITIAL_PROMPT = (
        "以下是台灣台語的對話，請使用台語漢字轉寫，"
        "保留台語用詞如：伊、啥物、佮、遮、彼。"
    )

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        model_dir: Optional[str] = None,
    ):
        """
        初始化辨識器。

        Parameters
        ----------
        model_size : str
            Whisper 模型大小。可選：tiny / base / small / medium / large-v3
            Fine-tuned 台語模型時，傳入 HuggingFace model id 或本地路徑。
        device : str
            "cuda" | "cpu" | "auto"（自動偵測）
        compute_type : str
            GPU: "float16" 或 "int8_float16"（省 VRAM）
            CPU: "int8"
        model_dir : str | None
            本地模型快取目錄，None 則使用 HuggingFace 預設快取。
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        print(f"[ASR] 正在載入模型 {model_size}，裝置：{device}，精度：{compute_type}")
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=model_dir,
        )
        print("[ASR] 模型載入完成 ✓")

    def transcribe_sync(
        self,
        audio_path: str | Path,
        language: str = "zh",
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        同步辨識入口（適用於腳本環境）。

        Parameters
        ----------
        audio_path : str | Path
            WAV/MP3/FLAC 音檔路徑，Faster-Whisper 內部會重取樣至 16kHz。
        language : str
            強制指定語言，"zh" 涵蓋台語漢字輸出。
            若為 None，Whisper 會自動語言偵測（速度略慢）。
        task : str
            "transcribe"（辨識）或 "translate"（翻譯成英文）
        """
        t0 = time.perf_counter()

        # Faster-Whisper 的 transcribe() 傳回 (generator, info)
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=self._BEAM_SIZE,
            initial_prompt=self._INITIAL_PROMPT,
            vad_filter=True,        # 內建 Silero-VAD，過濾靜音片段
            vad_parameters=dict(
                min_silence_duration_ms=300,   # 靜音超過 300ms 才切斷
                speech_pad_ms=100,             # 語音片段前後各補 100ms
            ),
            word_timestamps=False,  # 台語 word level 較不可靠，暫時關閉
        )

        result_segments: List[TranscriptionSegment] = []
        for seg in segments_gen:
            # log_prob 通常介於 -1~0，轉換為 0~1 的信心分數
            confidence = float(np.exp(seg.avg_logprob))
            result_segments.append(
                TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    confidence=confidence,
                    language=info.language,
                )
            )

        full_text = "".join(s.text for s in result_segments)
        duration_ms = (time.perf_counter() - t0) * 1000

        return TranscriptionResult(
            segments=result_segments,
            full_text=full_text,
            language=info.language,
            duration_ms=duration_ms,
        )

    async def transcribe(
        self,
        audio_path: str | Path,
        language: str = "zh",
    ) -> TranscriptionResult:
        """
        非同步辨識入口（適用於 FastAPI / LangGraph 整合）。

        將同步的 transcribe_sync 包裝在 ThreadPoolExecutor 中，
        避免 blocking event loop。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # 使用預設 ThreadPoolExecutor
            self.transcribe_sync,
            audio_path,
            language,
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        chunk_duration_s: float = 3.0,
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        串流辨識（適用於 WebSocket 即時語音輸入）。

        將麥克風的連續音訊切成 chunk_duration_s 秒的片段，
        逐片段送給 Whisper 辨識，實現低延遲視覺回饋。

        注意：此為 sliding-window 策略，對台語連音有截斷風險，
        第四階段整合 WebSocket 時可改用 VAD 切割 + 滾動 buffer。
        """
        buffer = np.array([], dtype=np.float32)
        chunk_samples = int(sample_rate * chunk_duration_s)

        async for chunk in audio_chunks:
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= chunk_samples:
                segment_audio = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                # 暫存到臨時 WAV 再送給 Whisper（Faster-Whisper 接受 ndarray 輸入）
                tmp_path = Path("/tmp/tomni_chunk.wav")
                sf.write(str(tmp_path), segment_audio, sample_rate)

                result = await self.transcribe(tmp_path)
                for seg in result.segments:
                    yield seg
