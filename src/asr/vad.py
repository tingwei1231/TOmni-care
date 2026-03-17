"""
src/asr/vad.py
==============
基於 Silero-VAD 的語音端點偵測（Voice Activity Detection）模組。

為什麼需要 VAD？
  長者說話節奏較慢，句與句之間沉默較多。若直接將含大量靜音的音訊
  送給 Whisper，會產生「幻覺（hallucination）」— 在靜音段輸出莫名文字。
  VAD 先切出有效語音片段，可大幅提升台語辨識準確度。

Silero-VAD 選用原因：
  - 輕量（~1MB），可在 CPU 即時運行
  - 支援 16kHz 單聲道輸入（與 Whisper 格式一致）
  - 對台語、閩南語等非英語有穩定表現
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class SpeechSegment:
    """VAD 偵測到的語音片段（單位：秒）。"""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SileroVAD:
    """
    Silero-VAD 包裝器。

    使用方式
    --------
    >>> vad = SileroVAD()
    >>> segments = vad.detect(audio_array, sample_rate=16000)
    >>> for seg in segments:
    ...     print(f"{seg.start:.2f}s ~ {seg.end:.2f}s")
    """

    # Silero-VAD 判斷為語音的機率門檻（0.5 為官方建議值，長者語速慢可適當降低）
    _THRESHOLD = 0.45
    # 最短有效語音長度（ms），低於此值視為噪音
    _MIN_SPEECH_DURATION_MS = 250
    # 靜音容忍長度（ms），短暫停頓不切斷語音段
    _MIN_SILENCE_DURATION_MS = 300

    def __init__(self, device: str = "cpu"):
        """
        載入 Silero-VAD 模型。

        使用 torch.hub 自動下載，首次需要網路連線。
        模型會被快取到 ~/.cache/torch/hub/。
        """
        self.device = device
        print("[VAD] 正在載入 Silero-VAD 模型...")
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model = self._model.to(device)
        # 解包官方工具函式
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self._utils
        print("[VAD] Silero-VAD 載入完成 ✓")

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[SpeechSegment]:
        """
        對音訊陣列進行 VAD 偵測。

        Parameters
        ----------
        audio : np.ndarray
            float32, shape (samples,)，數值範圍 [-1, 1]
        sample_rate : int
            Silero-VAD 僅支援 8000 或 16000 Hz

        Returns
        -------
        List[SpeechSegment]
            偵測到的語音時間段列表
        """
        # Silero-VAD 需要 torch.Tensor 輸入
        audio_tensor = torch.from_numpy(audio).to(self.device)

        timestamps = self.get_speech_timestamps(
            audio_tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self._THRESHOLD,
            min_speech_duration_ms=self._MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=self._MIN_SILENCE_DURATION_MS,
            return_seconds=True,   # 直接返回秒數，不需手動換算
        )

        return [
            SpeechSegment(start=ts["start"], end=ts["end"])
            for ts in timestamps
        ]

    def extract_speech_audio(
        self,
        audio: np.ndarray,
        segments: List[SpeechSegment],
        sample_rate: int = 16000,
        padding_ms: int = 100,
    ) -> List[np.ndarray]:
        """
        依照 VAD 結果裁切出語音片段。

        padding_ms：在片段前後各補 100ms，避免切掉語音起始的爆破音
        （台語字首的入聲韻尾如 -p/-t/-k 特別容易被截斷）
        """
        padding = int(sample_rate * padding_ms / 1000)
        clips = []
        for seg in segments:
            start_idx = max(0, int(seg.start * sample_rate) - padding)
            end_idx = min(len(audio), int(seg.end * sample_rate) + padding)
            clips.append(audio[start_idx:end_idx])
        return clips

    async def detect_async(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[SpeechSegment]:
        """非同步包裝版本，避免 blocking FastAPI event loop。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.detect, audio, sample_rate
        )


class StreamingVAD:
    """
    串流 VAD（用於 WebSocket 即時音訊輸入）。

    使用 Silero-VAD 的 VADIterator，對連續音訊 chunk 做即時語音端點偵測。
    適合第四階段 FastAPI WebSocket 整合。
    """

    _CHUNK_SIZE = 512  # 16kHz 下約 32ms，Silero-VAD 官方建議值

    def __init__(self, vad: SileroVAD, sample_rate: int = 16000):
        self._vad = vad
        self._sample_rate = sample_rate
        self._iterator = vad.VADIterator(
            model=vad._model,
            sampling_rate=sample_rate,
            threshold=SileroVAD._THRESHOLD,
            min_silence_duration_ms=SileroVAD._MIN_SILENCE_DURATION_MS,
        )
        self._speech_buffer: List[np.ndarray] = []

    def process_chunk(
        self, chunk: np.ndarray
    ) -> Tuple[bool, List[np.ndarray]]:
        """
        處理一個音訊 chunk。

        Returns
        -------
        (is_speech_ended, speech_chunks)
            is_speech_ended: True 表示一段語音說完了，可以送去 ASR
            speech_chunks: 完整語音片段的 numpy 陣列列表
        """
        chunk_tensor = torch.from_numpy(chunk)
        speech_dict = self._iterator(chunk_tensor, return_seconds=False)

        if speech_dict:
            # VADIterator 返回 {"start": ...} 或 {"end": ...}
            if "start" in speech_dict:
                self._speech_buffer = [chunk]
            elif "end" in speech_dict and self._speech_buffer:
                finished_chunks = self._speech_buffer.copy()
                self._speech_buffer = []
                return True, finished_chunks
        elif self._speech_buffer:
            # 語音進行中，持續累積 buffer
            self._speech_buffer.append(chunk)

        return False, []

    def reset(self):
        """重置狀態（每次對話開始時呼叫）。"""
        self._iterator.reset_states()
        self._speech_buffer = []
