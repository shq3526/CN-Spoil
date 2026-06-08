# CN-Spoil: Multimodal Chinese Clickbait Spoiling

## From Clickbait to Answers: Spoiler Generation for Multimodal Chinese News Feeds

Official implementation of **From Clickbait to Answers: Spoiler Generation for Multimodal Chinese News Feeds**, accepted at **ECML PKDD 2026 (Applied Data Science Track)**.

## Overview

CN-Spoil is a Chinese benchmark for clickbait spoiler generation. It contains 9,373 human-verified samples collected from major Chinese news platforms.

The project provides:

- the released textual annotations and dataset splits;
- the Gated Dual-Encoder implementation described in the paper;
- training and evaluation scripts;
- character-level evaluation with ROUGE, BLEU, METEOR, ChrF++, and BERTScore.

The paper studies a multimodal setting in which some spoilers require information from the cover image. The original cover images are not redistributed in this repository because they are third-party copyrighted material.

## Released Data

Each public CSV file uses the following columns:

```text
id,split,label,title,content,spoiler,source,category
```

The public release does not include:

- original article URLs;
- cover images;
- public image paths.

The `url` field was removed because source pages may change or disappear and because the public release should not depend on redistributing third-party source metadata. Cover images are omitted because the repository does not have redistribution rights for the original news images.

The `label` column preserves the benchmark label used by the released dataset. Its original encoding is retained.

## Image Manifest

The multimodal model requires authorized local copies of the cover images. Image paths are supplied through a separate local file:

```text
datasets/image_manifest.csv
```

Format:

```csv
id,image_path
cnspoil_train_00001,/absolute/path/to/image_00001.jpg
```

`image_manifest.csv` is intentionally not included in the public repository. The training and evaluation scripts merge it with the public CSV files using `id`.

Without authorized images, users can inspect and process the released textual annotations, but they cannot reproduce the multimodal training and evaluation reported in the paper.

## Dataset Statistics

| Split | Samples |
| --- | ---: |
| Train | 7,620 |
| Validation | 847 |
| Test | 906 |
| Total | 9,373 |

Additional statistics reported in the paper:

- average headline length: 25.8 characters;
- average article length: 3,159 characters;
- average spoiler length: 50.0 characters;
- visual-dependent samples: 29.0%.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- Evaluate
- pandas
- NumPy
- Pillow
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

Required pretrained models:

| Component | Checkpoint |
| --- | --- |
| Chinese BART | `fnlp/bart-base-chinese` |
| Chinese CLIP | `OFA-Sys/chinese-clip-vit-base-patch16` |

## Directory Structure

```text
CN-Spoil/
├── datasets/
│   ├── dataset_train.csv
│   ├── dataset_dev.csv
│   ├── dataset_test.csv
│   ├── README_DATA.md
│   └── image_manifest.csv
├── metrics/
│   ├── rouge/
│   ├── bleu/
│   ├── meteor/
│   ├── chrf/
│   └── bertscore/
├── models/
│   ├── bart-base-chinese/
│   └── chinese-clip-vit-base-patch16/
├── train_gated_bart_clip_cn.py
├── test_gated_bart_clip_cn.py
├── requirements.txt
└── README.md
```

`image_manifest.csv` and the image files are local resources and should not be committed.

## Training

Set the model, dataset, and image-manifest paths in `train_gated_bart_clip_cn.py`, then run:

```bash
python train_gated_bart_clip_cn.py
```

The checkpoint is saved to:

```text
./spoiler_bart_gated_cn
```

The script follows the paper configuration:

- frozen Chinese-CLIP ViT-B/16;
- Chinese BART generator;
- context-aware scalar gate;
- AdamW with learning rate `3e-5`;
- 10% warmup;
- weight decay `1e-2`;
- 10 epochs;
- random seed 42.

## Evaluation

Run:

```bash
python test_gated_bart_clip_cn.py
```

The script reports:

- ROUGE-1;
- ROUGE-L;
- BLEU;
- METEOR;
- ChrF++;
- BERTScore;
- average batch-size-1 inference latency.

Predictions and gate values are saved to:

```text
dataset_test_gated_cn_result.csv
```

## Main Results

| Model | Parameters | ROUGE-L |
| --- | ---: | ---: |
| BART (Text Only) | 0.14B | 49.10 |
| BART + CN-CLIP (Concat) | 0.23B | 58.84 |
| Gated Dual-Encoder | 0.23B | 64.55 |

Visual-dependent subset:

| Model | ROUGE-L |
| --- | ---: |
| BART (Text Only) | 8.34 |
| Gated Dual-Encoder | 74.97 |

These values are the results reported in the paper. The public repository does not include the copyrighted cover images required to rerun the multimodal experiments.

## Data Availability and Limitations

The public release contains the structured textual fields and human-verified spoiler annotations listed above. Original article URLs and cover images are not redistributed. This restriction prevents direct public reproduction of the image-dependent experiments, but avoids presenting third-party copyrighted material as part of the dataset.

The article text and annotations are released for academic research use only. Users are responsible for complying with applicable copyright rules, source-platform terms, and institutional data-governance requirements.

## License

- Code: Apache-2.0
- Dataset annotations and released research data: Research Use Only

## Acknowledgements

This project builds on Hugging Face Transformers, Chinese BART, and Chinese CLIP.
