# 必要なライブラリのインストール（ターミナルで実行）
# pip install datasets accelerate transformers[torch] librosa evaluate jiwer gradio whisper pandas pyarrow==14.0.1 ja-ginza sortedcontainers spacy
# pip install tensorboard
# pip install tensorboardX


import pandas as pd
import numpy as np
from datasets import Dataset, Audio, DatasetDict
from transformers import WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import spacy
import ginza

# CSVファイルのパス
file_path = './train0.csv'  # ローカルのCSVファイルのパスに変更

# CSVファイルを読み込む
df = pd.read_csv(file_path)

# 学習データと検証データを分ける
msk = np.random.rand(len(df)) < 0.7
train_dataset = Dataset.from_pandas(df[msk]).cast_column("path", Audio(sampling_rate=16000)).rename_column("path", "audio").remove_columns(["sampling_rate"])
validate_dataset = Dataset.from_pandas(df[~msk]).cast_column("path", Audio(sampling_rate=16000)).rename_column("path", "audio").remove_columns(["sampling_rate"])

# データセットをCSVファイルに保存
train_df = train_dataset.to_pandas()
validate_df = validate_dataset.to_pandas()

train_df.to_csv('./train.csv', index=False)  # 保存先のパスに変更
validate_df.to_csv('./validate.csv', index=False)  # 保存先のパスに変更

# データセットの辞書を作成
datasets = DatasetDict({
    "train": train_dataset,
    "validate": validate_dataset
})

# Whisperプロセッサのロード
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="Japanese", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["correct"]).input_ids
    return batch

prepared_datasets = datasets.map(prepare_dataset, num_proc=1)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 日本語評価用のセットアップ
nlp = spacy.load("ja_ginza")
ginza.set_split_mode(nlp, "C")
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [" ".join([str(i) for i in nlp(j)]) for j in pred_str]
    label_str = [" ".join([str(i) for i in nlp(j)]) for j in label_str]
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 訓練引数の設定
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-ja",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=5,
    max_steps=40,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10,
    eval_steps=10,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# モデルのロード
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")#("./models")
model.generation_config.language = "ja"

# Trainerの定義
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=prepared_datasets["train"],
    eval_dataset=prepared_datasets["validate"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# 学習を開始
trainer.train()

# モデルの保存
trainer.save_model("./tuning_models")  # 保存先のパスに変更
processor.save_pretrained("./tuning_models")  # 保存先のパスに変更