"""
src/tts/tone_sandhi.py
======================
台語連讀變調（Tone Sandhi）規則實作。

【語言學背景】
台語有 7 個基本聲調（第 2 聲與第 6 聲相同，實際 6 種）加上入聲（-p/-t/-k/-h 結尾）：

  聲調 | 調號 | 調型（數字）| 範例
  -----|------|------------|-----
    1  |  一平 |    44      | 東 (tang1)
    2  |  二上 |    53      | 黨 (tóng2)
    3  |  三去 |    31      | 棟 (tòng3)
    4  |  四入 |    32      | 篤 (tok4)  <- 短促，-p/-t/-k結尾
    5  |  五上 |   214      | 洞 (tōng5)
    7  |  七去 |    33      | (無常用例)
    8  |  八入 |     5      | 毒 (to̍k8)  <- 短促，高平

【連讀變調規則】（台語口語核心）
  非最後一個音節（non-final syllable）會依照規則變調：

  原調 → 變調
   1   →  7
   7   →  3
   3   →  2
   2   →  1
   5   →  7  （部份腔口為 3）
   4   →  8  （入聲變調）
   8   →  4  （入聲變調）

  例：「食飯」(tsia̍h-pn̄g)：
    食 (tsia̍h8) → 變調 → tsia̍h4（因接續音節）
    飯 (pn̄g7)   → 不變（final position）

實作說明：
  本模組接收 POJ（教會羅馬字）或 TLs（台灣閩南語羅馬字拼音）格式，
  解析聲調數字，應用連讀變調，再輸出變調後的拼音序列給 TTS 使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── 變調對照表 ────────────────────────────────────────────────
# 非入聲（舒聲）：1→7, 7→3, 3→2, 2→1, 5→7
# 入聲（促聲）：4→8, 8→4
SANDHI_MAP: dict[int, int] = {
    1: 7,
    7: 3,
    3: 2,
    2: 1,
    5: 7,
    4: 8,
    8: 4,
}


# ── 入聲韻尾（Checked Tones）─────────────────────────────────
# 這些結尾的音節屬於入聲，用以判斷是否套用入聲變調規則
CHECKED_TONE_ENDINGS = re.compile(r"[ptkhPTKH]$")


@dataclass
class TLSyllable:
    """
    台灣閩南語拼音（TLs）音節的結構化表示。

    Attributes
    ----------
    romanized : str
        不含聲調數字的拼音主體，例如 "tsia̍h" → "tsiah"
    tone : int
        聲調數字 1-8（0 表示輕聲/中性調）
    original : str
        原始輸入字串，用於還原
    """

    romanized: str    # 拼音（不含調號數字）
    tone: int         # 聲調 1-8，0=輕聲
    original: str     # 原始字串

    @property
    def is_checked_tone(self) -> bool:
        """判斷是否為入聲（韻尾為 -p/-t/-k/-h）。"""
        return bool(CHECKED_TONE_ENDINGS.search(self.romanized.lower()))

    def with_tone(self, new_tone: int) -> str:
        """傳回套用新聲調數字後的拼音字串。"""
        return f"{self.romanized}{new_tone}"


def parse_tl_syllable(syllable_str: str) -> TLSyllable:
    """
    解析一個 TLs 音節字串，分離拼音主體與聲調數字。

    Parameters
    ----------
    syllable_str : str
        例如：'tsia̍h8', 'pn̄g7', 'tang1', 'a0'（輕聲）

    Returns
    -------
    TLSyllable
    """
    # 嘗試提取尾端的數字作為聲調
    match = re.match(r"^(.*?)([0-8])$", syllable_str.strip())
    if match:
        romanized = match.group(1)
        tone = int(match.group(2))
    else:
        # 無數字尾碼，視為中性調（通常是輕聲助詞如 "ê"、"á"）
        romanized = syllable_str.strip()
        tone = 0

    return TLSyllable(romanized=romanized, tone=tone, original=syllable_str)


def apply_tone_sandhi(syllables: List[TLSyllable]) -> List[TLSyllable]:
    """
    對音節序列套用台語連讀變調規則。

    規則：
      - 最後一個音節（final position）**不變調**
      - 其餘所有音節（non-final）依 SANDHI_MAP 變調
      - 輕聲（tone=0）不參與變調

    Parameters
    ----------
    syllables : List[TLSyllable]
        輸入音節列表（已解析）

    Returns
    -------
    List[TLSyllable]
        變調後的音節列表
    """
    if not syllables:
        return syllables

    result = []
    for i, syl in enumerate(syllables):
        is_final = i == len(syllables) - 1
        # 輕聲或最後一個音節不變調
        if is_final or syl.tone == 0:
            result.append(syl)
            continue

        new_tone = SANDHI_MAP.get(syl.tone, syl.tone)
        result.append(
            TLSyllable(
                romanized=syl.romanized,
                tone=new_tone,
                original=syl.original,
            )
        )

    return result


def process_phrase(phrase: str) -> Tuple[str, List[int]]:
    """
    對一個台語詞組（以空白分隔音節）進行完整變調處理。

    Parameters
    ----------
    phrase : str
        例如：'tsia̍h8 pn̄g7' 或 'tang1 png7 a5'

    Returns
    -------
    (changed_phrase, original_tones)
        changed_phrase : 變調後的拼音字串（空白分隔）
        original_tones : 原始聲調數字列表（供除錯）
    """
    tokens = phrase.strip().split()
    syllables = [parse_tl_syllable(t) for t in tokens]
    original_tones = [s.tone for s in syllables]

    after_sandhi = apply_tone_sandhi(syllables)
    changed_phrase = " ".join(s.with_tone(s.tone) for s in after_sandhi)

    return changed_phrase, original_tones


# ── 台語漢字→拼音對照表（小型範例，實際應從完整詞典載入）──────────
# 格式：{漢字: TLs拼音字串}
# 完整詞典建議使用教育部《臺灣閩南語常用詞辭典》資料集
HANZI_TO_TL_SAMPLE: dict[str, str] = {
    "食": "tsia̍h8",
    "飯": "pn̄g7",
    "真": "tsin1",
    "好": "hó2",
    "謝謝": "siā-siā0",
    "你": "lí2",
    "我": "guá2",
    "伊": "i1",
    "啥物": "siánn-mih0",
    "佮": "kah4",
    "遮": "tsia1",
    "彼": "hit4",
    "𪜶": "tshuì3",  # 嘴（台語漢字）
    "腹肚": "pak4-tōo2",
    "歹勢": "pháinn2-sè3",
    "無代誌": "bô5-tāi7-tsì3",
}


def hanzi_to_tl(text: str) -> Optional[str]:
    """
    將台語漢字句子轉換為 TLs 拼音（詞為單位，以空白分隔）。

    參數
    ----
    text : str
        台語漢字輸入，例如「食飯真好」

    注意：此函式為示範版本，僅使用小型詞典。
    生產環境建議整合 g2pM 或「台語文字轉換工具」API。
    """
    result_tokens: List[str] = []
    i = 0
    while i < len(text):
        matched = False
        # 嘗試最長匹配（優先二字詞）
        for length in [4, 3, 2, 1]:
            chunk = text[i : i + length]
            if chunk in HANZI_TO_TL_SAMPLE:
                result_tokens.append(HANZI_TO_TL_SAMPLE[chunk])
                i += length
                matched = True
                break
        if not matched:
            # 未知字直接保留（讓後端 TTS 嘗試處理）
            result_tokens.append(text[i])
            i += 1

    return " ".join(result_tokens) if result_tokens else None
