"""
src/emotion/feature_extractor.py — librosa MFCC/Pitch/Energy 特徵提取
詳見 c:\\Users\\User\\Desktop\\TOmni-care 版本，此為 d:\\TOmni-care 正式版。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class AudioFeatures:
    """
    音訊情緒特徵向量。
    feature_vector = MFCC(40) + delta(40) + delta2(40) + pitch_stats(5) + energy_stats(5) = 130 維
    """
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    pitch_stats: np.ndarray   # [mean, std, min, max, range]
    energy_stats: np.ndarray  # [mean, std, min, max, range]
    duration_s: float
    sample_rate: int

    @property
    def feature_vector(self) -> np.ndarray:
        return np.concatenate([self.mfcc, self.mfcc_delta, self.mfcc_delta2,
                               self.pitch_stats, self.energy_stats])

    @property
    def n_features(self) -> int:
        return len(self.feature_vector)

    def to_dict(self) -> Dict:
        return {
            "mfcc_mean": float(self.mfcc.mean()),
            "mfcc_std": float(self.mfcc.std()),
            "pitch_mean": float(self.pitch_stats[0]),
            "pitch_range": float(self.pitch_stats[4]),
            "energy_mean": float(self.energy_stats[0]),
            "energy_std": float(self.energy_stats[1]),
            "duration_s": self.duration_s,
            "n_features": self.n_features,
        }


class EmotionFeatureExtractor:
    """
    台語語音情緒特徵提取器（MFCC + Pitch/PYIN + RMS Energy）。

    台語特殊考量：
      - 7 個聲調本身引起 Pitch 變化，使用 MFCC delta 解耦聲調 Pitch 與情緒 Pitch
      - 長者語音說話速度慢，HOP_LENGTH=512（約 23ms）足夠時間解析度
    """
    N_MFCC = 40
    F0_MIN = 75    # 長者男聲最低頻率
    F0_MAX = 400   # 長者女聲最高頻率
    HOP_LENGTH = 512

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract(self, audio_input, sr: Optional[int] = None) -> AudioFeatures:
        """
        提取完整情緒特徵向量。

        Parameters
        ----------
        audio_input : str | Path | np.ndarray
            音訊檔案路徑或 float32 array
        sr : int | None
            ndarray 輸入時的取樣率
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("請安裝：pip install librosa")

        if isinstance(audio_input, (str, Path)):
            y, sr = librosa.load(str(audio_input), sr=self.sample_rate)
        else:
            y = audio_input.astype(np.float32)
            sr = sr or self.sample_rate
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate

        duration_s = len(y) / sr

        # MFCC：音色/聲道共振特徵，焦慮時高頻能量偏高
        mfcc_m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.N_MFCC, hop_length=self.HOP_LENGTH)
        mfcc_mean = mfcc_m.mean(axis=1)
        mfcc_delta = librosa.feature.delta(mfcc_m, order=1).mean(axis=1)
        mfcc_delta2 = librosa.feature.delta(mfcc_m, order=2).mean(axis=1)

        # Pitch/F0：PYIN 對台語聲調更穩健，使用 voiced_flag 過濾非語音段
        f0, voiced_flag, _ = librosa.pyin(y, fmin=self.F0_MIN, fmax=self.F0_MAX,
                                          sr=sr, hop_length=self.HOP_LENGTH)
        vf0 = f0[voiced_flag]
        if len(vf0) > 0:
            pitch_stats = np.array([vf0.mean(), vf0.std(), vf0.min(), vf0.max(), vf0.max()-vf0.min()])
        else:
            pitch_stats = np.zeros(5)

        # RMS Energy：能量不穩定 → 焦慮；持續低 → 悲傷
        rms = librosa.feature.rms(y=y, hop_length=self.HOP_LENGTH)[0]
        energy_stats = np.array([rms.mean(), rms.std(), rms.min(), rms.max(), rms.max()-rms.min()])

        return AudioFeatures(mfcc=mfcc_mean, mfcc_delta=mfcc_delta, mfcc_delta2=mfcc_delta2,
                             pitch_stats=pitch_stats, energy_stats=energy_stats,
                             duration_s=duration_s, sample_rate=sr)

    async def extract_async(self, audio_input, sr: Optional[int] = None) -> AudioFeatures:
        """非同步包裝（FastAPI / LangGraph Node 使用）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, audio_input, sr)

    def extract_realtime_stats(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        即時串流用的輕量版特徵（每 100ms 更新）。
        ZCR 高 → 子音密集/說話急促（焦慮）；低 → 母音主導/說話緩慢（平靜）。
        """
        import librosa
        rms = float(librosa.feature.rms(y=y).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        return {
            "energy_rms": rms,
            "zero_crossing_rate": zcr,
            "tension_index": min(1.0, zcr * 10 + rms * 5),
        }
