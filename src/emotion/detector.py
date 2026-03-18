"""
src/emotion/detector.py — 情緒偵測主入口
整合 EmotionFeatureExtractor + EmotionSVM，提供 async API 給 LangGraph。
降級：無 SVM 模型時用 pitch/energy 規則型 fallback。
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Optional
from .feature_extractor import EmotionFeatureExtractor
from .classifier import EmotionLabel, EmotionPrediction, EmotionSVM

_RULE = {"high_energy":0.06, "high_pitch_range":80.0, "low_energy":0.02}

def _rule_predict(features) -> EmotionPrediction:
    em = features.energy_stats[0]
    es = features.energy_stats[1]
    pr = features.pitch_stats[4]
    if pr > _RULE["high_pitch_range"] and em > _RULE["high_energy"]:
        label = EmotionLabel.ANXIOUS if es > 0.03 else EmotionLabel.ANGRY
    elif em < _RULE["low_energy"]:
        label = EmotionLabel.SAD
    else:
        label = EmotionLabel.CALM
    return EmotionPrediction(label=label, confidence=0.6,
                             probabilities={e.value:0.0 for e in EmotionLabel},
                             needs_comfort=label.needs_comfort)

class EmotionDetector:
    """
    情緒偵測器。
    confidence_threshold：信心度低於此值時保守預測為 CALM（避免誤觸安撫模式）。
    """
    def __init__(self, model_path=None, sample_rate=22050, confidence_threshold=0.55):
        self._extractor = EmotionFeatureExtractor(sample_rate=sample_rate)
        self._clf: Optional[EmotionSVM] = None
        self.confidence_threshold = confidence_threshold
        if model_path and Path(model_path).exists():
            self._clf = EmotionSVM.load(model_path)
            print("[Emotion] SVM 模型載入完成 ✓")
        else:
            print("[Emotion] ⚠ 未找到 SVM 模型，使用規則型 fallback")

    def detect_sync(self, audio_input, sr=None) -> EmotionPrediction:
        features = self._extractor.extract(audio_input, sr=sr)
        if self._clf:
            pred = self._clf.predict(features.feature_vector)
            if pred.confidence < self.confidence_threshold:
                pred = EmotionPrediction(label=EmotionLabel.CALM,
                                        confidence=pred.confidence,
                                        probabilities=pred.probabilities,
                                        needs_comfort=False)
        else:
            pred = _rule_predict(features)
        print(f"[Emotion] {pred}")
        return pred

    async def detect(self, audio_input, sr=None) -> EmotionPrediction:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect_sync, audio_input, sr)
