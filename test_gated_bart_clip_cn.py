import os
import time
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    BertTokenizer,
    ChineseCLIPImageProcessor,
    ChineseCLIPVisionModel,
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
    "model_dir": "./spoiler_bart_gated_cn",
    "bart_base": "fnlp/bart-base-chinese",
    "clip_base": "OFA-Sys/chinese-clip-vit-base-patch16",
    "bertscore_model": "./models/bart-base-chinese",
    "test_data_path": "./datasets/dataset_test.csv",
    "image_manifest_path": "./datasets/image_manifest.csv",
    "output_path": "dataset_test_gated_cn_result.csv",
    "max_input_len": 1024,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


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

    def generate(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        return_gate=False,
        **kwargs,
    ):
        fused_embeddings, fused_attention_mask, gate = self._fuse(
            input_ids,
            attention_mask,
            pixel_values,
        )
        sequences = self.bart.generate(
            inputs_embeds=fused_embeddings,
            attention_mask=fused_attention_mask,
            **kwargs,
        )
        if return_gate:
            return sequences, gate
        return sequences

    @classmethod
    def from_pretrained(
        cls,
        save_directory,
        bart_base,
        clip_base,
        map_location,
    ):
        model = cls(bart_base, clip_base)
        model.bart = BartForConditionalGeneration.from_pretrained(save_directory)
        model.visual_projection.load_state_dict(
            torch.load(
                os.path.join(save_directory, "visual_projection.bin"),
                map_location=map_location,
            )
        )
        model.gate_net.load_state_dict(
            torch.load(
                os.path.join(save_directory, "gate_net.bin"),
                map_location=map_location,
            )
        )
        return model


def load_data(data_path, image_manifest_path):
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
        dataframe["split"] != "test",
        "split",
    ].unique()
    if len(invalid_splits):
        raise ValueError(
            f"Unexpected split values in {data_path}: {invalid_splits.tolist()}"
        )

    manifest = pd.read_csv(image_manifest_path)
    if not {"id", "image_path"}.issubset(manifest.columns):
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

    return dataframe


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    set_seed(CONFIG["seed"])

    tokenizer_source = (
        CONFIG["model_dir"]
        if os.path.exists(
            os.path.join(CONFIG["model_dir"], "tokenizer_config.json")
        )
        else CONFIG["bart_base"]
    )
    processor_source = (
        CONFIG["model_dir"]
        if os.path.exists(
            os.path.join(CONFIG["model_dir"], "preprocessor_config.json")
        )
        else CONFIG["clip_base"]
    )

    tokenizer = BertTokenizer.from_pretrained(tokenizer_source)
    image_processor = ChineseCLIPImageProcessor.from_pretrained(
        processor_source
    )
    model = GatedMultimodalBART.from_pretrained(
        CONFIG["model_dir"],
        CONFIG["bart_base"],
        CONFIG["clip_base"],
        CONFIG["device"],
    )
    model.to(CONFIG["device"])
    model.eval()

    rouge = evaluate.load("./metrics/rouge")
    bleu = evaluate.load("./metrics/bleu")
    meteor = evaluate.load("./metrics/meteor")
    chrf = evaluate.load("./metrics/chrf")
    bertscore = evaluate.load("./metrics/bertscore")

    dataframe = load_data(
        CONFIG["test_data_path"],
        CONFIG["image_manifest_path"],
    )

    records = []
    raw_predictions = []
    raw_references = []
    character_predictions = []
    character_references = []
    latencies = []

    separator = tokenizer.sep_token or "[SEP]"

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        text_input = f'{row["title"]} {separator} {row["content"]}'

        inputs = tokenizer(
            text_input,
            max_length=CONFIG["max_input_len"],
            truncation=True,
            return_tensors="pt",
        ).to(CONFIG["device"])

        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Cover image not found: {image_path}")
        with Image.open(image_path) as image:
            pixel_values = image_processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            ).pixel_values.to(CONFIG["device"])

        synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs, gate = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=pixel_values,
                return_gate=True,
                max_new_tokens=100,
                min_length=1,
                num_beams=5,
                length_penalty=1.5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                repetition_penalty=1.2,
            )
        synchronize()
        latencies.append((time.perf_counter() - start_time) * 1000)

        prediction = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).replace(" ", "")
        reference = str(row["spoiler"])

        records.append(
            {
                "id": row["id"],
                "title": row["title"],
                "reference": reference,
                "prediction": prediction,
                "gate": float(gate.squeeze().detach().cpu()),
            }
        )
        raw_predictions.append(prediction)
        raw_references.append(reference)
        character_predictions.append(" ".join(prediction))
        character_references.append(" ".join(reference))

    pd.DataFrame(records).to_csv(
        CONFIG["output_path"],
        index=False,
        encoding="utf-8-sig",
    )

    rouge_result = rouge.compute(
        predictions=character_predictions,
        references=character_references,
    )
    bleu_result = bleu.compute(
        predictions=character_predictions,
        references=[[reference] for reference in character_references],
    )
    meteor_result = meteor.compute(
        predictions=character_predictions,
        references=character_references,
    )
    chrf_result = chrf.compute(
        predictions=character_predictions,
        references=character_references,
        word_order=2,
    )
    bertscore_result = bertscore.compute(
        predictions=character_predictions,
        references=character_references,
        lang="zh",
        model_type=CONFIG["bertscore_model"],
        num_layers=6,
    )

    scores = {
        "R-1": rouge_result["rouge1"] * 100,
        "R-L": rouge_result["rougeL"] * 100,
        "BLEU": bleu_result["bleu"] * 100,
        "METEOR": meteor_result["meteor"] * 100,
        "ChrF++": chrf_result["score"],
        "BERTScore": np.mean(bertscore_result["f1"]) * 100,
        "Latency(ms)": float(np.mean(latencies)),
    }

    for name, value in scores.items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
