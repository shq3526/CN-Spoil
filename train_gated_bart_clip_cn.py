"""
train_gated_bart_clip_cn.py: Training script for Gated Multimodal BART using Chinese CLIP.
Adaptive gating is used to fuse CLIP vision features with BART language features.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import evaluate
import nltk
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BartForConditionalGeneration,
    ChineseCLIPVisionModel,
    ChineseCLIPImageProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# === 1. Environment Configuration ===
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_DOWNLOAD_URL = "https://mirror.ghproxy.com/https://github.com/tensorflow/nmt/raw/master/nmt/scripts/bleu.py"
# Local path for NLTK data
nltk.data.path.insert(0, "/root/nltk_data")

# === 2. Global Configuration ===
CONFIG = {
    "bart_model_name": "./models/bart-base-chinese",
    "clip_model_name": "./models/chinese-clip-vit-base-patch16", 
    "train_path": "./datasets/dataset_train.csv",
    "dev_path":   "./datasets/dataset_dev.csv",
    "image_manifest_path": "./datasets/image_manifest.csv",
    "output_dir": "./spoiler_bart_gated_cn", # Output directory for the final model
    "max_input_len": 1024,
    "max_target_len": 128,
    "batch_size": 16,      
    "epochs": 10,
    "learning_rate": 3e-5,
}

print("📉 Loading evaluation metrics...")
rouge = evaluate.load("./metrics/rouge")
bleu = evaluate.load("./metrics/bleu")
meteor = evaluate.load("./metrics/meteor")

# === 3. Model Definition (Gated Version) ===
class GatedMultimodalBART(nn.Module):
    """
    Multimodal BART following the paper:
    text embeddings and the projected image vector jointly determine
    a single scalar visual gate g in (0, 1).
    """
    def __init__(self, bart_model_name, clip_model_name, tokenizer_len):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.clip = ChineseCLIPVisionModel.from_pretrained(clip_model_name)

        for param in self.clip.parameters():
            param.requires_grad = False

        hidden_size = self.bart.config.d_model
        self.visual_projection = nn.Linear(self.clip.config.hidden_size, hidden_size)

        # Paper definition: g = sigmoid(W_g [v ; h_text] + b_g)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

        safe_vocab_size = tokenizer_len + 128
        self.bart.resize_token_embeddings(safe_vocab_size)

        self.config = self.bart.config
        self.generation_config = self.bart.generation_config

    @staticmethod
    def _text_mean(inputs_embeds, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        return (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _fuse(self, input_ids, attention_mask, pixel_values):
        # H_text: token-level BART embeddings
        inputs_embeds = self.bart.model.shared(input_ids)
        h_text = self._text_mean(inputs_embeds, attention_mask)

        with torch.no_grad():
            image_features = self.clip(pixel_values).pooler_output

        # v: projected global visual token
        visual_vector = self.visual_projection(image_features)

        # A single scalar gate is computed from text and image together.
        gate_input = torch.cat((visual_vector, h_text), dim=-1)
        gate = self.gate_net(gate_input)                 # (Batch, 1)
        visual_embeds = (visual_vector * gate).unsqueeze(1)

        fused_embeds = torch.cat((visual_embeds, inputs_embeds), dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        fused_mask = torch.cat((visual_mask, attention_mask), dim=1)
        return fused_embeds, fused_mask

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, **kwargs):
        fused_embeds, fused_mask = self._fuse(
            input_ids, attention_mask, pixel_values
        )
        return self.bart(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            labels=labels,
            return_dict=True
        )

    def generate(self, input_ids, attention_mask, pixel_values, **kwargs):
        fused_embeds, fused_mask = self._fuse(
            input_ids, attention_mask, pixel_values
        )
        return self.bart.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            **kwargs
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.bart.save_pretrained(save_directory)
        torch.save(
            self.visual_projection.state_dict(),
            os.path.join(save_directory, "visual_projection.bin")
        )
        torch.save(
            self.gate_net.state_dict(),
            os.path.join(save_directory, "gate_net.bin")
        )

    @classmethod
    def from_pretrained(cls, save_directory, bart_base, clip_base):
        temp_tokenizer = BertTokenizer.from_pretrained(bart_base)
        model = cls(bart_base, clip_base, len(temp_tokenizer))
        model.bart = BartForConditionalGeneration.from_pretrained(save_directory)

        proj_path = os.path.join(save_directory, "visual_projection.bin")
        gate_path = os.path.join(save_directory, "gate_net.bin")

        if os.path.exists(proj_path):
            model.visual_projection.load_state_dict(
                torch.load(proj_path, map_location="cpu")
            )
        if os.path.exists(gate_path):
            model.gate_net.load_state_dict(
                torch.load(gate_path, map_location="cpu")
            )
        return model

# === 4. Dataset (Upgraded: Automated Fuzzy Image Search) ===
class SpoilerDataset(Dataset):
    """
    Dataset with automated recursive searching for missing image files.
    """
    def __init__(self, data_list, tokenizer, image_processor, max_input_len, max_target_len):
        self.data = data_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.vocab_limit = len(tokenizer)
        self.missing_streak = 0          # Concurrent missing count
        self.missing_threshold = 100     # Warning threshold
        self.has_warned_block = False
        
    def __len__(self): return len(self.data)

    def _load_image(self, img_path_str):

        try:
            raw_str = str(img_path_str).strip().strip('\r\n').strip("[]'\" ")
            
            if ',' in raw_str: target_path = raw_str.split(',')[0]
            elif '|' in raw_str: target_path = raw_str.split('|')[0]
            else: target_path = raw_str
            
            target_path = target_path.strip(" '\"")
            
            target_filename = os.path.basename(target_path)
            target_name_no_ext = os.path.splitext(target_filename)[0]
            
            search_roots = [
                "datasets", 
                "/root/autodl-tmp/datasets",
                os.path.dirname(target_path) if os.path.dirname(target_path) else "." 
            ]
            
            for root_dir in search_roots:
                if not os.path.exists(root_dir): continue
                
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        current_name_no_ext = os.path.splitext(file)[0]
                        
                        if current_name_no_ext == target_name_no_ext:
                            full_path = os.path.join(root, file)
                            
                            try:
                                img = Image.open(full_path).convert("RGB")
                                self._reset_warning()
                                return img
                            except:
                                continue
                                
        except Exception:
            pass

        # === Warning Logic ===
        self.missing_streak += 1
        if self.missing_streak >= self.missing_threshold and not self.has_warned_block:
            print(f"\n⚠️ [CRITICAL WARNING] Massive image data missing! (Consecutive count: {self.missing_streak})")
            # Log sample for troubleshooting
            print(f"   Sample filename: {target_name_no_ext} (Not found in 'datasets' directory)")
            self.has_warned_block = True
            
        return Image.new('RGB', (224, 224), color='black')

    def _reset_warning(self):
        if self.missing_streak > 0:
            self.missing_streak = 0
            self.has_warned_block = False

    def __getitem__(self, idx):
        """
        Retrieves a single training example, including image features and text prompt.
        """
        item = self.data[idx]
        title = str(item['title'])
        content = str(item['content'])
        spoiler = str(item['spoiler'])
        
        # Invoke robust image loader
        image = self._load_image(item.get('images', ''))
        
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze()
        
        # Note: Prompt is kept in Chinese to match model training requirements
        input_text = (
            f"角色：NLP数据标注专家\n"
            f"任务：根据【标题】悬念，从【正文】和【图片】中提取精准剧透。\n"
            f"分类标准与规范（严格执行）：\n"
            f"1. Phrase (短语)：若问实体（人名/地点/物品），仅输出核心词（<15字）。❌拒绝“是xx”、“答案是xx”。\n"
            f"2. Passage (段落)：若问原因/过程，提取逻辑完整的精简句（<80字）。\n"
            f"3. Multi (列表)：若涉及多个对象，必须用顿号“、”分隔。\n"
            f"4. 标注铁律：\n"
            f"   - 指代消解：把“他/这”替换为具体名称。\n"
            f"   - 去新闻腔：删除“小编觉得”、“令人震惊”等废话。\n"
            f"标题：{title}\n"
            f"正文：{content}\n"
            f"精准剧透："
        )

        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_input_len-1, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            text_target=spoiler, 
            max_length=self.max_target_len, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        input_ids = model_inputs["input_ids"].squeeze()
        labels_ids = labels["input_ids"].squeeze()
        
        # Vocab safety check and label masking
        input_ids[input_ids >= self.vocab_limit] = self.tokenizer.unk_token_id
        labels_ids[labels_ids >= self.vocab_limit] = self.tokenizer.unk_token_id
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids, 
            "attention_mask": model_inputs["attention_mask"].squeeze(), 
            "pixel_values": pixel_values, 
            "labels": labels_ids
        }

# === 5. Metric Calculation ===
def compute_metrics(eval_pred, tokenizer):
    """
    Computes ROUGE, BLEU, and METEOR scores for validation monitoring.
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds_char = [" ".join([c for c in p.replace(" ", "")]) for p in decoded_preds]
    decoded_labels_char = [" ".join([c for c in l.replace(" ", "")]) for l in decoded_labels]

    r = rouge.compute(predictions=decoded_preds_char, references=decoded_labels_char)
    b = bleu.compute(predictions=decoded_preds_char, references=[[l] for l in decoded_labels_char])
    m = meteor.compute(predictions=decoded_preds_char, references=decoded_labels_char)

    return {
        "rouge-l": round(r["rougeL"] * 100, 4),
        "bleu": round(b["bleu"] * 100, 4),
        "meteor": round(m["meteor"] * 100, 4)
    }

def load_data(path, image_manifest_path, expected_split):
    """
    Read the current public CN-Spoil format:
    id,split,label,title,content,spoiler,source,category

    Images are joined from image_manifest.csv:
    id,image_path
    """
    if not os.path.exists(path):
        return []
    if not os.path.exists(image_manifest_path):
        raise FileNotFoundError(
            f"Image manifest not found: {image_manifest_path}"
        )

    df = pd.read_csv(path, dtype={"id": str, "split": str}, on_bad_lines="skip")
    required = {
        "id", "split", "label", "title", "content",
        "spoiler", "source", "category"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df[df["split"].astype(str) == expected_split].copy()
    df = df.dropna(subset=["id", "title", "spoiler"])

    manifest = pd.read_csv(
        image_manifest_path,
        dtype={"id": str},
        on_bad_lines="skip"
    )
    if not {"id", "image_path"}.issubset(manifest.columns):
        raise ValueError(
            "image_manifest.csv must contain id,image_path"
        )

    manifest = manifest[["id", "image_path"]].drop_duplicates("id")
    df = df.merge(manifest, on="id", how="left", validate="one_to_one")
    df = df.rename(columns={"image_path": "images"})

    missing_images = df["images"].isna().sum()
    if missing_images:
        print(f"⚠️ {missing_images} samples have no image path in manifest.")

    return df.to_dict("records")

def train():
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bart_model_name'])
    img_proc = ChineseCLIPImageProcessor.from_pretrained(CONFIG['clip_model_name'])
    
    # Use Gated model variant
    model = GatedMultimodalBART(CONFIG['bart_model_name'], CONFIG['clip_model_name'], len(tokenizer))
    
    model.bart.generation_config.repetition_penalty = 1.2
    
    print("📖 Loading training dataset...")
    train_data = load_data(CONFIG['train_path'], CONFIG['image_manifest_path'], "train")
    print("📖 Loading validation dataset...")
    val_data = load_data(CONFIG['dev_path'], CONFIG['image_manifest_path'], "dev")

    # Basic safety check for empty data
    if not train_data or not val_data:
        raise ValueError("❌ Train or Validation dataset is empty. Check path configurations!")

    # Pass complete data lists directly to Dataset
    train_ds = SpoilerDataset(train_data, tokenizer, img_proc, CONFIG['max_input_len'], CONFIG['max_target_len'])
    val_ds = SpoilerDataset(val_data, tokenizer, img_proc, CONFIG['max_input_len'], CONFIG['max_target_len'])

    args = Seq2SeqTrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=2,
        dataloader_num_workers=0, 
        bf16=True, 
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG['learning_rate'],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rouge-l",
        predict_with_generate=True,
        generation_max_length=64,
        save_safetensors=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    print("🚀 Starting Gated model variant training...")
    trainer.train()
    
    print("💾 Saving model and processor...")
    model.save_pretrained(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])
    img_proc.save_pretrained(CONFIG['output_dir'])

if __name__ == "__main__":
    train()