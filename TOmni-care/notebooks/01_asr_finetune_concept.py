# ==========================================
# TOmni-Care ASR Fine-tuning 概念碼
# 01_asr_finetune_concept.ipynb → 轉換為 .py 版本
# 適合在 Colab T4 上執行
# ==========================================

"""
【TAT-Corpus 台語 ASR 微調概念碼】
TAT-Corpus（Taiwanese Across Taiwan）是國立台灣大學語言學研究所
建立的台語語音語料庫，涵蓋多種腔口（閩南話各地方言）。

Fine-tuning 流程：
  1. 資料準備（TAT-Corpus .wav + .txt 對）
  2. 使用 WhisperProcessor tokenize 文字與音訊
  3. 凍結 Encoder，僅微調 Decoder（節省 VRAM）
  4. 使用 Seq2SeqTrainer 訓練
  5. 匯出 CTranslate2 格式供 Faster-Whisper 使用

執行環境需求：
  - Colab T4 GPU（16GB VRAM）
  - faster-whisper, transformers, datasets, evaluate
"""

# ────────────────────────────────────────────────
# Colab 安裝指令（在 Cell 中執行）
# ────────────────────────────────────────────────
COLAB_INSTALL = """
!pip install -q faster-whisper transformers datasets evaluate jiwer
!pip install -q ctranslate2  # 用於匯出 CTranslate2 格式
"""

# ────────────────────────────────────────────────
# Step 1：資料集準備
# ────────────────────────────────────────────────
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def prepare_tat_dataset(corpus_dir: str) -> "DatasetDict":
    """
    從 TAT-Corpus 目錄建立 HuggingFace DatasetDict。

    TAT-Corpus 目錄結構（預期格式）：
      corpus_dir/
        train/
          audio_001.wav
          audio_001.txt    ← 對應台語漢字標注
          audio_002.wav
          audio_002.txt
        dev/
          ...

    若無 TAT-Corpus，可使用 Mozilla Common Voice 台語子集：
      from datasets import load_dataset
      dataset = load_dataset("mozilla-foundation/common_voice_16_0", "nan-tw")
    """
    from datasets import Audio, Dataset, DatasetDict

    def load_split(split_dir: Path) -> Dataset:
        wav_files = sorted(split_dir.glob("*.wav"))
        records = []
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            with open(txt_path, encoding="utf-8") as f:
                text = f.read().strip()
            records.append({"audio": str(wav_path), "sentence": text})

        dataset = Dataset.from_list(records)
        # 讓 HuggingFace 自動處理音訊重取樣到 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        return dataset

    corpus_path = Path(corpus_dir)
    train_dataset = load_split(corpus_path / "train")
    dev_dataset = load_split(corpus_path / "dev")

    print(f"[資料集] Train: {len(train_dataset)} 筆，Dev: {len(dev_dataset)} 筆")
    return DatasetDict({"train": train_dataset, "validation": dev_dataset})


# ────────────────────────────────────────────────
# Step 2：特徵提取器與 Tokenizer 初始化
# ────────────────────────────────────────────────
def init_processor(model_name: str = "openai/whisper-large-v3"):
    """
    初始化 Whisper Processor（包含 Feature Extractor + Tokenizer）。

    language="zh" 讓 Tokenizer 使用繁體中文詞彙表，
    台語漢字大部分與繁中重疊，因此此設定優於 "nan"（南語）。
    """
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="Chinese",     # 引導使用漢字詞彙表
        task="transcribe",
    )
    print(f"[Processor] 載入完成，詞彙表大小：{processor.tokenizer.vocab_size}")
    return processor


# ────────────────────────────────────────────────
# Step 3：資料集前處理（Tokenize + Feature Extraction）
# ────────────────────────────────────────────────
def preprocess_dataset(dataset, processor, max_label_length: int = 448):
    """
    將原始音訊 + 文字轉換為 Whisper 模型輸入格式。

    音訊前處理：
      - 重取樣至 16kHz（已由 HuggingFace Audio column 處理）
      - 30 秒定長 log-mel 頻譜（Whisper 固定輸入長度）

    文字前處理：
      - Tokenize 台語漢字文字
      - 截斷至 448 tokens（Whisper decoder 最大長度）
    """

    def process_batch(batch):
        # 取得音訊 numpy array（已是 16kHz float32）
        audio = batch["audio"]

        # 計算 log-mel 頻譜（80 維，30 秒定長，不足補零）
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        # Tokenize 台語文字
        batch["labels"] = processor.tokenizer(
            batch["sentence"],
            max_length=max_label_length,
            truncation=True,
        ).input_ids

        return batch

    processed = dataset.map(
        process_batch,
        remove_columns=["audio", "sentence"],
        num_proc=1,   # Colab 環境建議 num_proc=1
    )
    return processed


# ────────────────────────────────────────────────
# Step 4：Data Collator（動態 padding）
# ────────────────────────────────────────────────
@dataclass
class WhisperDataCollator:
    """
    Whisper 專用 Data Collator。

    Whisper 的輸入特徵（log-mel）已是固定長度，不需 padding。
    只有 labels（decoder 序列）需要動態 padding，
    並將 padding token 設為 -100（CrossEntropy loss 忽略）。
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 輸入特徵（固定 80×3000，直接 stack）
        input_features = torch.stack(
            [torch.tensor(f["input_features"]) for f in features]
        )

        # labels（可變長度，需 padding）
        label_features = [
            {"input_ids": f["labels"]} for f in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # 將 padding 位置替換為 -100，讓 loss 函式忽略這些位置
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 移除 decoder 起始 token（Whisper 訓練時不計算起始 token 的 loss）
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ────────────────────────────────────────────────
# Step 5：評估指標（CER — 字元錯誤率）
# ────────────────────────────────────────────────
def compute_metrics_factory(processor):
    """
    建立 CER（Character Error Rate）評估函式。

    臺語使用 CER 而非 WER（詞錯誤率），因為：
      - 台語詞界定困難（無空白分隔）
      - 細粒度的字元評估更能反映音素辨識準確度
    """
    import evaluate

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # 將 -100 替換回 pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # 解碼預測與標籤
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_metrics


# ────────────────────────────────────────────────
# Step 6：訓練設定與啟動
# ────────────────────────────────────────────────
def run_finetune(
    corpus_dir: str,
    output_dir: str = "./models/whisper-taiwanese",
    base_model: str = "openai/whisper-large-v3",
    num_epochs: int = 3,
    batch_size: int = 4,       # T4 16GB VRAM 下的安全批次大小
    learning_rate: float = 1e-5,
    freeze_encoder: bool = True,  # 凍結 encoder，僅微調 decoder（節省 VRAM）
):
    """
    完整 Fine-tuning 流程。

    freeze_encoder=True 的效果：
      - 訓練速度提升約 2x
      - VRAM 節省約 30%（不需儲存 encoder 梯度）
      - 在資料量較少（< 50 小時）時通常比全量訓練效果更好
    """
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
    )

    print("[訓練] 初始化模型與 Processor...")
    processor = init_processor(base_model)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)

    # 設定模型語言強制 token
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="zh", task="transcribe"
    )
    model.config.suppress_tokens = []

    if freeze_encoder:
        print("[訓練] 凍結 Encoder（只微調 Decoder）")
        model.model.encoder.requires_grad_(False)

    print("[訓練] 準備資料集...")
    raw_dataset = prepare_tat_dataset(corpus_dir)
    processed_dataset = preprocess_dataset(raw_dataset, processor)

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,       # 模擬 batch_size=8
        warmup_steps=200,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,                           # T4 GPU 開啟 FP16
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,             # CER 越低越好
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(processor),
        tokenizer=processor.feature_extractor,
    )

    print("[訓練] 開始 Fine-tuning...")
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)
    print(f"[訓練] 完成！模型儲存至：{output_dir}")


# ────────────────────────────────────────────────
# Step 7：匯出為 CTranslate2 格式（供 Faster-Whisper 使用）
# ────────────────────────────────────────────────
EXPORT_COMMAND = """
# 在訓練完成後執行此指令，將 HuggingFace 格式轉換為 CTranslate2 格式
# 轉換後模型約縮小 50%（float16），推論速度快 2~4x

ct2-transformers-converter \\
  --model ./models/whisper-taiwanese \\
  --output_dir ./models/whisper-taiwanese-ct2 \\
  --quantization float16 \\
  --copy_files tokenizer.json

# 使用轉換後的模型進行推論：
# from faster_whisper import WhisperModel
# model = WhisperModel("./models/whisper-taiwanese-ct2", device="cuda")
"""

if __name__ == "__main__":
    # 範例：使用 TAT-Corpus 進行微調
    # run_finetune(
    #     corpus_dir="./data/corpus/tat",
    #     output_dir="./models/whisper-taiwanese",
    #     freeze_encoder=True,
    # )
    print("請取消上方 run_finetune 的註解並設定 corpus_dir 後執行。")
    print("\n若使用 Common Voice 台語資料集，請修改 prepare_tat_dataset 函式：")
    print("  from datasets import load_dataset")
    print("  dataset = load_dataset('mozilla-foundation/common_voice_16_0', 'nan-tw')")
