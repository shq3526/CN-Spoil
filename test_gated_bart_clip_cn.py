"""
Inference and evaluation script for the Gated Multimodal BART model (CLIP-CN variant).
Calculates ROUGE, BLEU, METEOR, ChrF++, and BERTScore for evaluation.
"""

import os
import sys
# Configure HF endpoint for mirror access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import nltk
# Ensure NLTK data path is configured
nltk.data.path.insert(0, "/root/nltk_data")
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import evaluate
import re
from PIL import Image
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BartForConditionalGeneration,
    ChineseCLIPVisionModel,
    ChineseCLIPImageProcessor,
)
# 如果没有安装 sentence_transformers，这里可能会报错，建议 pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

# === 1. Configuration ===
CONFIG = {
    "model_dir": "./spoiler_bart_gated_cn", # Output directory from train_gated_cn_cilp.py
    
    # Base model checkpoints
    "bart_base": "fnlp/bart-base-chinese", 
    "clip_base": "OFA-Sys/chinese-clip-vit-base-patch16",
    
    "local_bart_path": "./models/bart-base-chinese", # Used for BERTScore calculation
    "sbert_path": "./models/paraphrase-multilingual-MiniLM-L12-v2",
    
    "test_data_path": "./datasets/dataset_test.csv",
    "image_manifest_path": "./datasets/image_manifest.csv",
    "output_path": "dataset_test_gated_cn_result.csv",
    "max_len": 1024,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# === 2. Model Structure (Gated) - Must match training ===
class GatedMultimodalBART(nn.Module):
    """
    Inference structure matching the paper and training script:
    text and image jointly compute one scalar visual gate.
    """
    def __init__(self, bart_model_name, clip_model_name):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.clip = ChineseCLIPVisionModel.from_pretrained(clip_model_name)

        for param in self.clip.parameters():
            param.requires_grad = False

        hidden_size = self.bart.config.d_model
        self.visual_projection = nn.Linear(
            self.clip.config.hidden_size, hidden_size
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _text_mean(inputs_embeds, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        return (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @classmethod
    def from_pretrained(cls, save_directory, bart_base, clip_base):
        model = cls(bart_base, clip_base)
        print(f"📥 Loading weights from {save_directory}...")
        model.bart = BartForConditionalGeneration.from_pretrained(save_directory)

        proj_path = os.path.join(save_directory, "visual_projection.bin")
        gate_path = os.path.join(save_directory, "gate_net.bin")

        if os.path.exists(proj_path):
            model.visual_projection.load_state_dict(
                torch.load(proj_path, map_location=CONFIG["device"])
            )
        if os.path.exists(gate_path):
            model.gate_net.load_state_dict(
                torch.load(gate_path, map_location=CONFIG["device"])
            )
            print("✅ Gating weights loaded successfully")
        else:
            print("❌ Warning: Gating weights not found!")
        return model

    def generate(self, input_ids, attention_mask, pixel_values, **kwargs):
        inputs_embeds = self.bart.model.shared(input_ids)
        h_text = self._text_mean(inputs_embeds, attention_mask)

        with torch.no_grad():
            image_features = self.clip(pixel_values).pooler_output

        visual_vector = self.visual_projection(image_features)
        gate_input = torch.cat((visual_vector, h_text), dim=-1)
        gate = self.gate_net(gate_input)
        visual_embeds = (visual_vector * gate).unsqueeze(1)

        fused_embeds = torch.cat((visual_embeds, inputs_embeds), dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        fused_mask = torch.cat((visual_mask, attention_mask), dim=1)

        return self.bart.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            **kwargs
        )

# === 3. Utility Functions ===
def load_image(img_path_str):
    """
    Robust image loader with recursive fuzzy searching:
    1. Extension-agnostic matching.
    2. Deep recursive search under the datasets root.
    """
    try:
        # 1. Clean path and extract filename stem
        raw_str = str(img_path_str).strip().strip('\r\n').strip("[]'\" ")
        
        if ',' in raw_str: target_path = raw_str.split(',')[0]
        elif '|' in raw_str: target_path = raw_str.split('|')[0]
        else: target_path = raw_str
        
        target_path = target_path.strip(" '\"")
        
        # Extract filename stem
        target_filename = os.path.basename(target_path)
        target_name_no_ext = os.path.splitext(target_filename)[0]
        
        # 2. Define search scopes
        search_roots = [
            "datasets", 
            "/root/autodl-tmp/datasets",
            os.path.dirname(target_path) if os.path.dirname(target_path) else "."
        ]
        
        # 3. Exhaustive search
        for root_dir in search_roots:
            if not os.path.exists(root_dir): continue
            
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    current_name_no_ext = os.path.splitext(file)[0]
                    
                    # Match by stem regardless of extension
                    if current_name_no_ext == target_name_no_ext:
                        full_path = os.path.join(root, file)
                        try:
                            return Image.open(full_path).convert("RGB")
                        except:
                            continue
    except Exception:
        pass
        
    return Image.new('RGB', (224, 224), color='black')

def extract_relevant_content(sbert_model, title, content, max_chars=800):
    """
    Extracts relevant context segments based on SBERT semantic similarity to the title.
    """
    sentences = re.split(r'(?<=[。？！?!])', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences or len(content) < max_chars: return content[:max_chars]
    try:
        title_emb = sbert_model.encode(title, convert_to_tensor=True, show_progress_bar=False)
        sent_embs = sbert_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(title_emb, sent_embs)[0]
        scored = [(i, scores[i].item(), sentences[i]) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = []
        curr_len = 0
        for idx, s, txt in scored:
            if curr_len + len(txt) < max_chars:
                selected.append(idx)
                curr_len += len(txt)
            else: break
        selected.sort()
        return "".join([sentences[i] for i in selected])
    except: return content[:max_chars]

# === 4. Main Pipeline ===
def main():
    print(f"🚀 Loading Gated Model: {CONFIG['model_dir']}")
    # Automatic initialization of Tokenizer and Processor
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bart_base'])
    img_proc = ChineseCLIPImageProcessor.from_pretrained(CONFIG['clip_base'])
    
    # Attempt SBERT loading
    try: 
        sbert = SentenceTransformer(CONFIG['sbert_path'])
    except: 
        print("⚠️ Local SBERT loading failed, attempting download...")
        sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 加载 Gated 模型
    model = GatedMultimodalBART.from_pretrained(CONFIG['model_dir'], CONFIG['bart_base'], CONFIG['clip_base'])
    model.to(CONFIG['device'])
    model.eval()

    print("📉 Loading metrics...")
    # Load local evaluation scripts
    rouge = evaluate.load("./metrics/rouge")
    bleu = evaluate.load("./metrics/bleu")
    meteor = evaluate.load("./metrics/meteor")
    chrf = evaluate.load("./metrics/chrf")
    bertscore = evaluate.load("./metrics/bertscore")

    # Read current CN-Spoil CSV format and join images by id.
    df = pd.read_csv(
        CONFIG["test_data_path"],
        dtype={"id": str, "split": str},
        on_bad_lines="skip"
    )
    required = {
        "id", "split", "label", "title", "content",
        "spoiler", "source", "category"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{CONFIG['test_data_path']} missing columns: {sorted(missing)}"
        )

    df = df[df["split"].astype(str) == "test"].copy()
    manifest = pd.read_csv(
        CONFIG["image_manifest_path"],
        dtype={"id": str},
        on_bad_lines="skip"
    )
    if not {"id", "image_path"}.issubset(manifest.columns):
        raise ValueError("image_manifest.csv must contain id,image_path")

    manifest = manifest[["id", "image_path"]].drop_duplicates("id")
    df = df.merge(manifest, on="id", how="left", validate="one_to_one")

    results, raw_preds, raw_refs, char_preds, char_refs = [], [], [], [], []

    print("⚡ Starting Gated Inference...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = str(row["title"])
        content = str(row["content"])
        ground_truth = str(row["spoiler"])
        images = str(row["image_path"])

        refined = extract_relevant_content(sbert, title, content, max_chars=CONFIG['max_len'] - 200)

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
            f"正文：{refined}\n"
            f"精准剧透："
        )

        inputs = tokenizer(input_text, max_length=CONFIG['max_len']-1, truncation=True, return_tensors="pt").to(CONFIG['device'])
        
        pil_img = load_image(images)
        pixel_values = img_proc(images=pil_img, return_tensors="pt").pixel_values.to(CONFIG['device'])

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=pixel_values,
                max_new_tokens=100,
                min_length=1,
                num_beams=5,
                length_penalty=1.5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                repetition_penalty=1.2
            )

        spoiler = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
        results.append({"Title": title, "Ground_Truth": ground_truth, "Generated_Spoiler": spoiler})

        raw_preds.append(spoiler)
        raw_refs.append(ground_truth)
        char_preds.append(" ".join([c for c in spoiler]))
        char_refs.append(" ".join([c for c in ground_truth]))

    # Save results
    pd.DataFrame(results).to_csv(CONFIG['output_path'], index=False, encoding='utf-8-sig')
    print(f"✅ Predictions saved to: {CONFIG['output_path']}")

    print("\n📊 Evaluation Report (LaTeX formatted outputs included):")
    
    # === Calculate 6 core metrics ===
    try:
        r = rouge.compute(predictions=char_preds, references=char_refs)
        b = bleu.compute(predictions=char_preds, references=[[ref] for ref in char_refs])
        m = meteor.compute(predictions=char_preds, references=char_refs)
        c = chrf.compute(predictions=raw_preds, references=raw_refs)
        bs = bertscore.compute(predictions=raw_preds, references=raw_refs, lang="zh", model_type=CONFIG['local_bart_path'], num_layers=6)

        # Formatting scores for LaTeX table compatibility
        r1_score = r['rouge1'] * 100
        rl_score = r['rougeL'] * 100
        bleu_score = b['bleu'] * 100
        meteor_score = m['meteor'] * 100
        chrf_score = c['score']
        bs_score = np.mean(bs['f1']) * 100

        print("-" * 50)
        print(f"R-1\t:\t{r1_score:.2f}")
        print(f"R-L\t:\t{rl_score:.2f}")
        print(f"BLEU\t:\t{bleu_score:.2f}")
        print(f"METEOR\t:\t{meteor_score:.2f}")
        print(f"ChrF++\t:\t{chrf_score:.2f}")
        print(f"BS\t:\t{bs_score:.2f}")
        print("-" * 50)
        
        # LaTeX formatted line for direct copying
        print("LaTeX formatted result:")
        print(f"{r1_score:.2f} & {rl_score:.2f} & {bleu_score:.2f} & {meteor_score:.2f} & {chrf_score:.2f} & {bs_score:.2f} \\\\")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ Error during metric calculation: {e}")
        print("💡 Tip: If Java is missing, run: apt-get update && apt-get install -y default-jre")

if __name__ == "__main__":
    main()