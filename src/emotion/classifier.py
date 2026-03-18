"""
src/emotion/classifier.py — SVM 情緒分類器

情緒類別（台語長者照護情境）：
  CALM（平靜）   → 一般閒聊、資訊詢問
  ANXIOUS（焦慮）→ 身體不適、說話急促
  SAD（悲傷）    → 孤寂、語調低沉
  ANGRY（煩躁）  → 不耐煩、抱怨

LangGraph 決策：needs_comfort=True → 切換安撫節點
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class EmotionLabel(str, Enum):
    CALM = "calm"
    ANXIOUS = "anxious"
    SAD = "sad"
    ANGRY = "angry"

    @property
    def chinese(self) -> str:
        return {"calm":"平靜","anxious":"焦慮","sad":"悲傷","angry":"煩躁"}[self.value]

    @property
    def needs_comfort(self) -> bool:
        """是否需要 LangGraph 切換至安撫模式。"""
        return self in (EmotionLabel.ANXIOUS, EmotionLabel.SAD, EmotionLabel.ANGRY)


@dataclass
class EmotionPrediction:
    label: EmotionLabel
    confidence: float
    probabilities: Dict[str, float]
    needs_comfort: bool

    def __str__(self):
        return f"{self.label.chinese}({self.label.value}) 信心 {self.confidence:.1%}"


class EmotionSVM:
    """
    SVM 情緒分類器。

    SVM 選用理由：長者照護資料量小（<1000 筆），SVM 在小資料集泛化能力
    優於深度學習；RBF kernel 對 MFCC 特徵效果最佳。
    """

    def __init__(self, kernel="rbf", C=10.0, gamma="scale"):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self._model = None
        self._scaler = None
        self._classes: List[str] = []

    def train(self, X: np.ndarray, y: List[str], test_size=0.2) -> Dict:
        """
        訓練流程：
          1. StandardScaler 正規化（SVM 對尺度敏感）
          2. 訓練 SVM with probability=True（Platt scaling）
          3. class_weight='balanced' 處理類別不平衡
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        self._scaler = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)

        self._model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,
                          probability=True, class_weight="balanced", random_state=42)
        self._model.fit(X_train_s, y_train)
        self._classes = list(self._model.classes_)

        y_pred = self._model.predict(X_test_s)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[Emotion] 訓練完成 ✓ 準確率：{accuracy:.1%}")
        return {"accuracy": accuracy, "report": classification_report(y_test, y_pred)}

    def predict(self, feature_vector: np.ndarray) -> EmotionPrediction:
        if self._model is None:
            raise RuntimeError("模型未訓練，請先呼叫 train() 或 load()")
        x = self._scaler.transform(feature_vector.reshape(1, -1))
        proba = self._model.predict_proba(x)[0]
        pred_str = self._model.predict(x)[0]
        return EmotionPrediction(
            label=EmotionLabel(pred_str),
            confidence=float(proba.max()),
            probabilities=dict(zip(self._classes, proba.tolist())),
            needs_comfort=EmotionLabel(pred_str).needs_comfort,
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model":self._model,"scaler":self._scaler,
                         "classes":self._classes,"kernel":self.kernel,
                         "C":self.C,"gamma":self.gamma}, f)

    @classmethod
    def load(cls, path) -> "EmotionSVM":
        with open(path, "rb") as f:
            s = pickle.load(f)
        clf = cls(kernel=s["kernel"], C=s["C"], gamma=s["gamma"])
        clf._model, clf._scaler, clf._classes = s["model"], s["scaler"], s["classes"]
        return clf


def generate_synthetic_training_data(n_samples_per_class=50) -> Tuple[np.ndarray, List[str]]:
    """
    合成訓練資料（無真實標注資料時用於 pipeline 測試）。
    各情緒的特徵分布依語音情緒文獻設定：
      calm: 低 energy、低 pitch_range
      anxious: 高 pitch_range、能量不穩定
      sad: 低 pitch-mean、低 energy
      angry: 高 energy、高 pitch-mean
    ⚠ 僅供測試，生產環境需真實標注台語語音資料。
    """
    rng = np.random.default_rng(42)
    n_features = 130  # 40+40+40+5+5
    params = {"calm":(0.0,0.5),"anxious":(1.5,0.8),"sad":(-1.0,0.6),"angry":(2.0,1.0)}
    X_list, y_list = [], []
    for label, (mean, std) in params.items():
        X_list.append(rng.normal(mean, std, (n_samples_per_class, n_features)))
        y_list.extend([label]*n_samples_per_class)
    return np.vstack(X_list), y_list
