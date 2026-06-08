import os
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BartForConditionalGeneration,
    BertTokenizer,
    ChineseCLIPImageProcessor,
    ChineseCLIPVisionModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PUBLIC_COLUMNS = [
    "id",
    "split",
    "label",
    "title",
    "content",
    "spoiler",
    "source",
    "category",
]

CONFIG = {
    "bart_model_name": "./models/bart-base-chinese",
    "clip_model_name": "./models/chinese-clip-vit-base-patch16",
    "train_path": "./datasets/dataset_train.csv",
    "dev_path": "./datasets/dataset_dev.csv",
    "image_manifest_path": "./datasets/image_manifest.csv",
    "output_dir": "./spoiler_bart_gated_cn",
    "max_input_len": 1024,
    "max_target_len": 128,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 3e-5,
    "warmup_ratio": 0.10,
    "weight_decay": 1e-2,
    "seed": 42,
}

rouge = evaluate.load("./metrics/rouge")
bleu = evaluate.load("./metrics/bleu")
meteor = evaluate.load("./metrics/meteor")


class GatedMultimodalBART(nn.Module):
    def __init__(self, bart_model_name, clip_model_name):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.clip = ChineseCLIPVisionModel.from_pretrained(clip_model_name)

        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        text_dim = self.bart.config.d_model
        visual_dim = self.clip.config.hidden_size
        self.visual_projection = nn.Linear(visual_dim, text_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * text_dim, 1),
            nn.Sigmoid(),
        )

        self.config = self.bart.config
        self.generation_config = self.bart.generation_config

    def _fuse(self, input_ids, attention_mask, pixel_values):
        text_embeddings = self.bart.model.shared(input_ids)

        self.clip.eval()
        with torch.no_grad():
            raw_visual = self.clip(pixel_values=pixel_values).pooler_output

        visual_vector = self.visual_projection(raw_visual)
        text_mask = attention_mask.unsqueeze(-1).to(text_embeddings.dtype)
        text_summary = (text_embeddings * text_mask).sum(dim=1)
        text_summary = text_summary / text_mask.sum(dim=1).clamp_min(1.0)

        gate = self.gate_net(torch.cat([visual_vector, text_summary], dim=-1))
        visual_token = (gate * visual_vector).unsqueeze(1)

        fused_embeddings = torch.cat([visual_token, text_embeddings], dim=1)
        visual_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        fused_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return fused_embeddings, fused_attention_mask, gate

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels=None,
        **kwargs,
    ):
        fused_embeddings, fused_attention_mask, _ = self._fuse(
            input_ids,
            attention_mask,
            pixel_values,
        )
        return self.bart(
            inputs_embeds=fused_embeddings,
            attention_mask=fused_attention_mask,
            labels=labels,
            return_dict=True,
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.bart.save_pretrained(save_directory)
        torch.save(
            self.visual_projection.state_dict(),
            os.path.join(save_directory, "visual_projection.bin"),
        )
        torch.save(
            self.gate_net.state_dict(),
            os.path.join(save_directory, "gate_net.bin"),
        )


class SpoilerDataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        image_processor,
        max_input_len,
        max_target_len,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        separator = self.tokenizer.sep_token or "[SEP]"
        text_input = f'{item["title"]} {separator} {item["content"]}'

        model_inputs = self.tokenizer(
            text_input,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            text_target=str(item["spoiler"]),
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        image_path = Path(str(item["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Cover image not found: {image_path}")
        with Image.open(image_path) as image:
            pixel_values = self.image_processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            ).pixel_values.squeeze(0)

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "labels": labels,
        }


def load_data(data_path, image_manifest_path, expected_split):
    dataframe = pd.read_csv(data_path)
    missing_columns = [c for c in PUBLIC_COLUMNS if c not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")

    dataframe = dataframe[PUBLIC_COLUMNS].copy()
    dataframe = dataframe.dropna(
        subset=["id", "split", "title", "content", "spoiler"]
    )

    if dataframe["id"].duplicated().any():
        raise ValueError(f"Duplicate ids found in {data_path}")

    invalid_splits = dataframe.loc[
        dataframe["split"] != expected_split,
        "split",
    ].unique()
    if len(invalid_splits):
        raise ValueError(
            f"Unexpected split values in {data_path}: {invalid_splits.tolist()}"
        )

    manifest = pd.read_csv(image_manifest_path)
    required_manifest_columns = {"id", "image_path"}
    if not required_manifest_columns.issubset(manifest.columns):
        raise ValueError("image_manifest.csv must contain id,image_path")

    if manifest["id"].duplicated().any():
        raise ValueError("Duplicate ids found in image_manifest.csv")

    dataframe = dataframe.merge(
        manifest[["id", "image_path"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    missing_images = dataframe["image_path"].isna()
    if missing_images.any():
        ids = dataframe.loc[missing_images, "id"].head(10).tolist()
        raise ValueError(f"Missing image paths for ids: {ids}")

    return dataframe.to_dict("records")


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    character_predictions = [
        " ".join(text.replace(" ", "")) for text in decoded_predictions
    ]
    character_labels = [
        " ".join(text.replace(" ", "")) for text in decoded_labels
    ]

    rouge_result = rouge.compute(
        predictions=character_predictions,
        references=character_labels,
    )
    bleu_result = bleu.compute(
        predictions=character_predictions,
        references=[[reference] for reference in character_labels],
    )
    meteor_result = meteor.compute(
        predictions=character_predictions,
        references=character_labels,
    )

    return {
        "rouge-l": rouge_result["rougeL"] * 100,
        "bleu": bleu_result["bleu"] * 100,
        "meteor": meteor_result["meteor"] * 100,
    }


def main():
    set_seed(CONFIG["seed"])

    tokenizer = BertTokenizer.from_pretrained(CONFIG["bart_model_name"])
    image_processor = ChineseCLIPImageProcessor.from_pretrained(
        CONFIG["clip_model_name"]
    )
    model = GatedMultimodalBART(
        CONFIG["bart_model_name"],
        CONFIG["clip_model_name"],
    )

    train_dataset = SpoilerDataset(
        load_data(
            CONFIG["train_path"],
            CONFIG["image_manifest_path"],
            "train",
        ),
        tokenizer,
        image_processor,
        CONFIG["max_input_len"],
        CONFIG["max_target_len"],
    )
    validation_dataset = SpoilerDataset(
        load_data(
            CONFIG["dev_path"],
            CONFIG["image_manifest_path"],
            "dev",
        ),
        tokenizer,
        image_processor,
        CONFIG["max_input_len"],
        CONFIG["max_target_len"],
    )

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=2,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        optim="adamw_torch",
        seed=CONFIG["seed"],
        data_seed=CONFIG["seed"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rouge-l",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=64,
        save_safetensors=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda output: compute_metrics(output, tokenizer),
    )
    trainer.train()

    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    image_processor.save_pretrained(CONFIG["output_dir"])


if __name__ == "__main__":
    main()
