"""
src/tts/synthesizer.py
======================
基於 Bert-VITS2 的台語語音合成模組。

【Bert-VITS2 架構說明】
Bert-VITS2 以 VITS（Variational Inference with adversarial learning for TTS）
為基礎，加入 BERT 作為文字理解骨幹，讓合成語音更自然、情感更豐富。

台語 TTS 流程：
  輸入文字（漢字/TLs）
    → TextNormalizer（正規化）
    → ToneSandhi（連讀變調）
    → G2P（文字→音素）
    → BERT Encoder（取得語意向量）
    → VITS Decoder（生成波形）
    → 輸出 WAV

【模型下載】
台語 Bert-VITS2 預訓練模型：
  HuggingFace: j-min/bert-vits2-taiwanese（非官方，請確認授權）
  或使用 Facebook MMS-TTS（urn:facebook/mms-1b-fl102，支援台語）

【降級方案】
若 Bert-VITS2 模型尚未備妥，本模組提供降級至
  Google TTS（gtts）的 fallback 機制，確保開發期間可測試。
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ── 延遲導入（避免在無 GPU 環境下 import 失敗）─────────────────
def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class TaiwanTTS:
    """
    台語語音合成器。

    支援兩種後端（依環境自動切換）：
      1. Bert-VITS2（高品質，需 GPU + 模型檔案）
      2. gTTS fallback（低品質，無需 GPU，用於開發測試）

    使用方式
    --------
    >>> tts = TaiwanTTS(model_path="/models/tw_bert_vits2", device="cuda")
    >>> wav = await tts.synthesize("食飯真好！")
    >>> # wav 為 np.ndarray，shape (samples,)，16kHz float32
    """

    _SAMPLE_RATE = 22050  # Bert-VITS2 預設輸出取樣率

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
        speaker_id: int = 0,
        speed: float = 0.85,   # 台語給長者使用，語速稍慢
    ):
        """
        Parameters
        ----------
        model_path : str | None
            Bert-VITS2 模型目錄（含 G_*.pth 與 config.json）
            None → 自動降級至 gTTS fallback
        config_path : str | None
            None → 自動在 model_path 目錄下尋找 config.json
        device : str
            "cuda" | "cpu" | "auto"
        speaker_id : int
            多說話人模型時指定說話人（台語女聲建議 0）
        speed : float
            語速縮放比例（0.5~2.0，1.0 為原速）
        """
        self.device = self._resolve_device(device)
        self.speaker_id = speaker_id
        self.speed = speed
        self.sample_rate = self._SAMPLE_RATE

        # ── 模組依賴的延遲導入 ──────────────────────────────────
        from .text_normalizer import normalize
        from .tone_sandhi import process_phrase

        self._normalize = normalize
        self._process_phrase = process_phrase

        # ── 模型載入 ─────────────────────────────────────────
        self._model = None
        self._hps = None
        self._backend = "gtts"  # 預設降級模式

        if model_path and Path(model_path).exists():
            self._load_bert_vits2(model_path, config_path)
        else:
            print(
                "[TTS] ⚠ 未找到 Bert-VITS2 模型，切換至 gTTS fallback 模式。\n"
                "      生產環境請至 HuggingFace 下載台語預訓練模型。"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        torch = _try_import_torch()
        if device == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _load_bert_vits2(self, model_path: str, config_path: Optional[str]):
        """
        載入 Bert-VITS2 模型。

        架構說明：
          - config.json 包含模型超參數（採樣率、說話人數等）
          - G_*.pth 為 Generator 模型權重（VITS 解碼器）
          - 需要 bert/ 子目錄存放 BERT 模型權重
        """
        try:
            import torch
            # 這裡使用佔位符，實際整合需 clone Bert-VITS2 repo
            # 並 import 其 utils, models 模組
            # from bert_vits2 import utils, models

            config_file = config_path or str(
                Path(model_path) / "config.json"
            )

            # 尋找最新的 Generator checkpoint
            model_dir = Path(model_path)
            ckpt_files = sorted(model_dir.glob("G_*.pth"))
            if not ckpt_files:
                raise FileNotFoundError(f"找不到 G_*.pth 在 {model_path}")

            latest_ckpt = str(ckpt_files[-1])
            print(f"[TTS] 載入 Bert-VITS2 checkpoint: {latest_ckpt}")

            # --- 以下為 Bert-VITS2 實際整合程式碼框架 ---
            # hps = utils.get_hparams_from_file(config_file)
            # net_g = models.SynthesizerTrn(
            #     len(hps.symbols),
            #     hps.data.filter_length // 2 + 1,
            #     hps.train.segment_size // hps.data.hop_length,
            #     n_speakers=hps.data.n_speakers,
            #     **hps.model,
            # ).to(self.device)
            # utils.load_checkpoint(latest_ckpt, net_g, None)
            # net_g.eval()
            # self._model = net_g
            # self._hps = hps
            # self._backend = "bert_vits2"
            # print(f"[TTS] Bert-VITS2 載入完成 ✓，說話人 ID: {self.speaker_id}")

            # TODO: 取消上方註解並安裝 bert_vits2 套件後啟用
            raise NotImplementedError("Bert-VITS2 整合待完成，目前使用 gTTS fallback")

        except Exception as e:
            print(f"[TTS] Bert-VITS2 載入失敗（{e}），切換至 gTTS fallback")

    def _synthesize_bert_vits2(self, text: str) -> np.ndarray:
        """
        使用 Bert-VITS2 合成語音（Bert-VITS2 載入後才呼叫）。

        內部流程：
          1. 文字 → phoneme（使用台語 G2P）
          2. phoneme → BERT embedding
          3. BERT embedding + phoneme → VITS decoder → waveform
        """
        import torch

        # --- Bert-VITS2 實際推論程式碼框架 ---
        # from bert_vits2 import commons, utils
        # from bert_vits2.text import text_to_sequence
        #
        # # G2P：文字轉音素序列
        # phones, tones, lang_ids = text_to_sequence(text, self._hps)
        # phones = commons.intersperse(phones, 0)
        # tones = commons.intersperse(tones, 0)
        # lang_ids = commons.intersperse(lang_ids, 0)
        #
        # x = torch.LongTensor(phones).unsqueeze(0).to(self.device)
        # x_tones = torch.LongTensor(tones).unsqueeze(0).to(self.device)
        # x_lang = torch.LongTensor(lang_ids).unsqueeze(0).to(self.device)
        # x_lengths = torch.LongTensor([len(phones)]).to(self.device)
        # sid = torch.LongTensor([self.speaker_id]).to(self.device)
        #
        # with torch.no_grad():
        #     audio = self._model.infer(
        #         x, x_lengths, sid=sid,
        #         noise_scale=0.667, noise_scale_w=0.8,
        #         length_scale=1.0 / self.speed
        #     )[0][0, 0].cpu().numpy()
        # return audio
        raise NotImplementedError

    def _synthesize_gtts_fallback(self, text: str) -> np.ndarray:
        """
        gTTS fallback 合成（無需 GPU，用於開發/測試）。

        注意：gTTS 不支援台語，此 fallback 為中文 TTS。
        僅用於確認音訊 pipeline 流程正確，不作為產品語音使用。
        """
        try:
            from gtts import gTTS
            import soundfile as sf
        except ImportError:
            raise ImportError("請安裝 gTTS：pip install gTTS soundfile")

        tts_obj = gTTS(text=text, lang="zh-TW")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        tts_obj.save(mp3_path)

        # 使用 librosa 將 MP3 轉為 numpy array
        try:
            import librosa
            audio, _ = librosa.load(mp3_path, sr=self.sample_rate)
        finally:
            os.unlink(mp3_path)

        return audio.astype(np.float32)

    def synthesize_sync(
        self,
        text: str,
        apply_tone_sandhi: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        同步合成入口。

        Parameters
        ----------
        text : str
            輸入台語文字（漢字或 TLs 拼音）
        apply_tone_sandhi : bool
            是否套用連讀變調（正常對話應為 True）

        Returns
        -------
        (audio_array, sample_rate)
            audio_array : np.ndarray, float32, shape (samples,)
            sample_rate : int, 22050 Hz
        """
        t0 = time.perf_counter()

        # Step 1: 文字正規化（分句）
        sentences = self._normalize(text)

        all_audio_chunks = []
        short_silence = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)

        for sentence in sentences:
            # Step 2: 連讀變調（台語語音合成的關鍵步驟）
            if apply_tone_sandhi:
                # 若輸入含 TLs 拼音，先做變調
                sentence_for_tts = sentence  # 實際整合時傳入拼音序列
            else:
                sentence_for_tts = sentence

            # Step 3: 合成
            if self._backend == "bert_vits2":
                chunk = self._synthesize_bert_vits2(sentence_for_tts)
            else:
                chunk = self._synthesize_gtts_fallback(sentence_for_tts)

            all_audio_chunks.append(chunk)
            all_audio_chunks.append(short_silence)  # 句間添加短暫停頓

        audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[TTS] 合成完成 {elapsed_ms:.0f}ms，{len(audio)/self.sample_rate:.2f}s 音訊，後端：{self._backend}")

        return audio, self.sample_rate

    async def synthesize(
        self,
        text: str,
        apply_tone_sandhi: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        非同步合成入口（FastAPI 整合用）。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.synthesize_sync, text, apply_tone_sandhi
        )

    async def synthesize_to_bytes(
        self,
        text: str,
        format: str = "wav",
    ) -> bytes:
        """
        合成後直接返回音訊 bytes（WebSocket 串流傳輸用）。

        Parameters
        ----------
        format : str
            "wav"（推薦，無損）或 "mp3"
        """
        import soundfile as sf

        audio, sr = await self.synthesize(text)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format=format.upper())
        return buf.getvalue()
