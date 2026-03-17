"""
tests/test_tone_sandhi.py
=========================
台語連讀變調規則單元測試。

不需 GPU 即可執行，快速驗證語言學規則的正確性。
執行：pytest tests/test_tone_sandhi.py -v
"""

import pytest
from src.tts.tone_sandhi import (
    SANDHI_MAP,
    TLSyllable,
    apply_tone_sandhi,
    parse_tl_syllable,
    process_phrase,
)


class TestParseTLSyllable:
    def test_basic_parse(self):
        syl = parse_tl_syllable("tang1")
        assert syl.romanized == "tang"
        assert syl.tone == 1

    def test_parse_with_special_chars(self):
        """台語拼音含特殊 Unicode 字母（如 ̍）"""
        syl = parse_tl_syllable("tsia̍h8")
        assert syl.tone == 8
        assert syl.romanized == "tsia̍h"

    def test_neutral_tone(self):
        """輕聲（助詞等）無調號數字，解析為 tone=0"""
        syl = parse_tl_syllable("ê")
        assert syl.tone == 0

    def test_checked_tone_detection(self):
        """入聲（-p/-t/-k/-h 結尾）偵測"""
        assert parse_tl_syllable("tsia̍h8").is_checked_tone is True
        assert parse_tl_syllable("tok4").is_checked_tone is True
        assert parse_tl_syllable("tang1").is_checked_tone is False


class TestToneSandhi:
    """
    連讀變調規則測試。

    依據台語音韻學：
      一二三四五七八 → 七三二一七八四（舒聲）
                           入聲：四字第八調；八字第四調
    """

    @pytest.mark.parametrize(
        "input_tone, expected_tone",
        [
            (1, 7),  # 第一調 → 第七調
            (7, 3),  # 第七調 → 第三調
            (3, 2),  # 第三調 → 第二調
            (2, 1),  # 第二調 → 第一調
            (5, 7),  # 第五調 → 第七調
            (4, 8),  # 第四調（入聲）→ 第八調
            (8, 4),  # 第八調（入聲）→ 第四調
        ],
    )
    def test_sandhi_map(self, input_tone, expected_tone):
        assert SANDHI_MAP[input_tone] == expected_tone

    def test_final_syllable_unchanged(self):
        """最後一個音節不變調"""
        syllables = [
            TLSyllable("tang", 1, "tang1"),
            TLSyllable("png", 7, "png7"),  # final position
        ]
        result = apply_tone_sandhi(syllables)
        # 第一個（非 final）應變調
        assert result[0].tone == SANDHI_MAP[1]  # 1 → 7
        # 最後一個不變調
        assert result[1].tone == 7

    def test_single_syllable_unchanged(self):
        """單音節詞不變調"""
        syllables = [TLSyllable("tang", 1, "tang1")]
        result = apply_tone_sandhi(syllables)
        assert result[0].tone == 1

    def test_neutral_tone_unchanged(self):
        """輕聲不參與變調"""
        syllables = [
            TLSyllable("tang", 1, "tang1"),
            TLSyllable("a", 0, "a"),   # 輕聲
        ]
        result = apply_tone_sandhi(syllables)
        assert result[1].tone == 0  # 輕聲不變

    def test_eat_rice_example(self):
        """「食飯」(tsia̍h8 pn̄g7) 經典範例"""
        phrase = "tsia̍h8 pn̄g7"
        changed, originals = process_phrase(phrase)
        tokens = changed.split()
        # 食（8）→ 變調 4（入聲變調 8→4）
        assert tokens[0].endswith("4")
        # 飯（7）→ final，不變調
        assert tokens[1].endswith("7")


class TestProcessPhrase:
    def test_three_syllable_phrase(self):
        """三音節詞組，前兩個變調，最後一個不變調"""
        phrase = "tang1 si7 bo5"
        changed, originals = process_phrase(phrase)
        tokens = changed.split()
        assert originals == [1, 7, 5]
        # tang1 → 7（非 final）
        assert tokens[0].endswith("7")
        # si7 → 3（非 final）
        assert tokens[1].endswith("3")
        # bo5 → 不變（final）
        assert tokens[2].endswith("5")
