# CN-Spoil: Multimodal Chinese Clickbait Spoiling

### From Clickbait to Answers: Spoiler Generation for Multimodal Chinese News Feeds

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch\&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![ECML PKDD](https://img.shields.io/badge/ECML_PKDD-2026-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-green)

Official implementation of:

**From Clickbait to Answers: Spoiler Generation for Multimodal Chinese News Feeds**
Accepted at **ECML PKDD 2026 (Applied Data Science Track)**

---

## 📖 Overview

CN-Spoil is the first large-scale benchmark for multimodal Chinese clickbait spoiling.

The project introduces:

* **CN-Spoil Dataset**

  * 9,373 real-world Chinese news samples
  * Human-verified spoiler annotations
  * Visual-dependency labels
  * Collected from major Chinese news platforms

* **Gated Dual-Encoder**

  * Chinese-BART text generator
  * Chinese-CLIP visual encoder
  * Adaptive gating mechanism
  * Lightweight deployment-oriented architecture

Our analysis shows that nearly **29%** of clickbait spoilers require visual grounding and cannot be recovered reliably from article text alone.

---

## 🛠️ Prerequisites

| Component | Recommendation           |
| --------- | ------------------------ |
| Python    | 3.10+                    |
| PyTorch   | 2.0+                     |
| GPU       | NVIDIA GPU (recommended) |
| CUDA      | 11.8+                    |
| OS        | Linux / Windows / macOS  |

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/CN-Spoil.git
cd CN-Spoil
```

### Create Environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Pretrained Models

Please download the following pretrained checkpoints into the `models/` directory.

```bash
mkdir models
```

| Component    | HuggingFace Repository                                      |
| ------------ | ----------------------------------------------------------- |
| Chinese BART | fnlp/bart-base-chinese                                      |
| Chinese CLIP | OFA-Sys/chinese-clip-vit-base-patch16                       |
| SBERT        | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |

---

## 📂 Directory Structure

```text
CN-Spoil/
├── datasets/
│   ├── dataset_train.csv
│   ├── dataset_dev.csv
│   └── dataset_test.csv
│
├── metrics/
│   ├── rouge/
│   ├── bleu/
│   ├── meteor/
│   ├── chrf/
│   └── bertscore/
│
├── models/
│
├── spoiler_bart_gated_cn/
│
├── train_gated_bart_clip_cn.py
├── test_gated_bart_clip_cn.py
│
├── requirements.txt
└── README.md
```

---

## 🏋️ Training

Train the Gated Dual-Encoder model:

```bash
python train_gated_bart_clip_cn.py
```

The trained model will be saved to:

```text
./spoiler_bart_gated_cn
```

---

## 📊 Evaluation

Run inference and evaluation on the test set:

```bash
python test_gated_bart_clip_cn.py
```

Reported metrics include:

* ROUGE-1
* ROUGE-L
* BLEU
* METEOR
* ChrF++
* BERTScore

Results will be saved as:

```text
dataset_test_gated_cn_result.csv
```

---

## 📈 Main Results

| Model                   | Parameters | ROUGE-L |
| ----------------------- | ---------- | ------- |
| BART (Text Only)        | 0.14B      | 49.10   |
| BART + CN-CLIP (Concat) | 0.23B      | 58.84   |
| Gated Dual-Encoder      | 0.23B      | 64.55   |

Visual-dependent subset:

| Model              | ROUGE-L |
| ------------------ | ------- |
| BART (Text Only)   | 8.34    |
| Gated Dual-Encoder | 74.97   |

---

## 📄 Dataset

The CN-Spoil dataset contains:

* 9,373 Chinese news samples
* Headlines
* Article bodies
* Cover images
* Human-verified spoilers
* Visual dependency annotations

### Dataset Release

The dataset is released for academic research purposes only.

Please cite our paper when using CN-Spoil in your research.

---

---

## 📜 License

Code: Apache-2.0 License

Dataset: Research Use Only

---

## 🙏 Acknowledgements

This project builds upon:

* HuggingFace Transformers
* Chinese-BART
* Chinese-CLIP
* Sentence Transformers

We thank the ECML PKDD reviewers for their valuable feedback.

