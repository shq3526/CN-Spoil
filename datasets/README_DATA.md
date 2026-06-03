# CN-Spoil Dataset

## Overview

CN-Spoil is a benchmark for multimodal Chinese clickbait spoiling.

The dataset consists of real-world Chinese news articles collected from major news aggregation platforms and annotated with spoiler content and visual dependency labels.

The released version contains structured metadata, article content, spoiler annotations, and source information.

---

## Statistics

| Split      | Samples |
| ---------- | ------: |
| Train      |   7,620 |
| Validation |     847 |
| Test       |     906 |
| Total      |   9,373 |

Additional statistics:

* Average headline length: 25.8 characters
* Average article length: 3,159 characters
* Average spoiler length: 50.0 characters
* Visual-dependent samples: 29.0%

---

## File Structure

```text
datasets/
├── dataset_train.csv
├── dataset_dev.csv
├── dataset_test.csv
└── README_DATA.md
```

---

## Data Format

Each row contains the following fields:

| Column   | Description                        |
| -------- | ---------------------------------- |
| id       | Unique sample identifier           |
| split    | Dataset split (train / dev / test) |
| label    | Original clickbait label           |
| title    | News headline                      |
| content  | Full article body                  |
| spoiler  | Annotated spoiler                  |
| url      | Original article URL               |
| source   | News platform                      |
| category | News category                      |

Example:

```csv
id,split,label,title,content,spoiler,url,source,category
cnspoil_train_00001,train,1,神秘嘉宾终于现身...,正文内容...,周杰伦,https://example.com/news/123,Toutiao,Entertainment
```

---

## Visual Dependency Labels

CN-Spoil contains both text-sufficient and visual-dependent samples.

### Text-Sufficient

The spoiler can be inferred directly from the article content.

### Visual-Dependent

The spoiler requires information from the associated image and cannot be reliably recovered from article text alone.

Visual-dependent samples account for approximately 29% of the benchmark.

---

## News Sources

Articles were collected from publicly accessible Chinese news aggregation platforms, including:

* Jiemian News
* Toutiao
* Baidu News

---

## Annotation Pipeline

The annotation workflow consists of three stages:

1. Candidate article collection and filtering
2. Candidate spoiler generation
3. Human verification and refinement

All spoiler annotations were manually reviewed before inclusion in the dataset.
