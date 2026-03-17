"""
src/tts/text_normalizer.py
==========================
台語文字正規化前處理模組。

TTS 輸入往往夾雜多種格式：
  - 台語漢字（傳統漢字書寫）
  - POJ 拼音（教會羅馬字，例：Tâi-oân）
  - TLs 拼音（台灣閩南語羅馬字，例：Tâi-uân）
  - 標點符號與數字

本模組負責在送給 Bert-VITS2 之前，將各種輸入統一轉換為
「漢字 + TLs 拼音混排」格式，並做基本的文字清理。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# POJ → TLs 聲母/韻母對照（核心轉換差異）
_POJ_TO_TLS_MAP: dict[str, str] = {
    # 聲母對照
    "ch": "ts",     # POJ 的 "ch" 等於 TLs 的 "ts"
    "chh": "tsh",   # POJ 的 "chh" 等於 TLs 的 "tsh"
    "kh": "kh",     # 相同，無需轉換
    # 韻母差異
    "oe": "ue",     # POJ oe → TLs ue
    "eng": "ing",   # 部份方言
    "oan": "uann",  # POJ oan → TLs uann（鼻化）
    "oai": "uai",
}


def normalize_punctuation(text: str) -> str:
    """將全形標點轉為半形，並移除 TTS 無意義的特殊符號。"""
    # 全形轉半形
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("…", "...")
    text = text.replace("「", "")
    text = text.replace("」", "")
    # 移除多餘空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_poj_to_tls(poj_text: str) -> str:
    """
    將 POJ（教會羅馬字）轉換為 TLs（台灣閩南語羅馬字）。

    POJ 是傳教士制定的舊式羅馬字，TLs 是台灣教育部現行標準。
    Bert-VITS2 台語版本使用 TLs 作為訓練格式。

    轉換規則（依長度降序優先匹配，避免子字串衝突）：
    """
    # 依鍵長度降序排列，長的先匹配避免 "chh" 被 "ch" 提前截斷
    for poj, tls in sorted(
        _POJ_TO_TLS_MAP.items(), key=lambda x: len(x[0]), reverse=True
    ):
        poj_text = poj_text.replace(poj, tls)
    return poj_text


def split_to_chunks(text: str) -> List[str]:
    """
    將長文切成 TTS 適合的句子單位（以標點分句）。

    Bert-VITS2 在過長輸入時音質下降，
    建議每段不超過 50 個字元（台語音節約 25 個）。
    """
    # 以句尾標點分割
    sentences = re.split(r"([。！？.!?])", text)
    chunks: List[str] = []
    buffer = ""
    for token in sentences:
        buffer += token
        if re.search(r"[。！？.!?]", token) or len(buffer) > 50:
            stripped = buffer.strip()
            if stripped:
                chunks.append(stripped)
            buffer = ""
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def normalize(text: str, input_format: str = "auto") -> List[str]:
    """
    完整正規化流程：清理 → 格式轉換 → 分句。

    Parameters
    ----------
    text : str
        原始輸入文字
    input_format : str
        "auto" | "hanzi" | "poj" | "tls"

    Returns
    -------
    List[str]
        可直接送給 TTS 的句子列表
    """
    # Step 1: 統一 Unicode 正規形式（處理各種組合字母）
    text = unicodedata.normalize("NFC", text)

    # Step 2: 標點正規化
    text = normalize_punctuation(text)

    # Step 3: 格式轉換
    if input_format in ("poj", "auto"):
        # auto 模式：若包含 ch/chh 等 POJ 特徵，嘗試轉換
        if re.search(r"\bch[h]?\b", text):
            text = convert_poj_to_tls(text)

    # Step 4: 分句
    return split_to_chunks(text)
