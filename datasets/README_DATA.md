# CN-Spoil Dataset

## Overview

CN-Spoil is a benchmark for Chinese clickbait spoiler generation. The full research setting is multimodal, combining article text with a cover image. The public release contains the structured textual fields and spoiler annotations, while the original cover images are not redistributed because they are third-party copyrighted material.

## Statistics

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

## Files

```text
datasets/
├── dataset_train.csv
├── dataset_dev.csv
├── dataset_test.csv
├── image_manifest.example.csv
└── README_DATA.md
```

The public CSV files contain textual data only. `image_manifest.example.csv` documents the format expected for authorized local images.

## Public CSV Schema

The column order is:

```text
id,split,label,title,content,spoiler,source,category
```

| Column | Description |
| --- | --- |
| `id` | Stable unique identifier used to join local multimodal resources |
| `split` | Dataset split: `train`, `dev`, or `test` |
| `label` | Released benchmark label; the original encoding is retained |
| `title` | News headline |
| `content` | Article body included in the research release |
| `spoiler` | Human-verified spoiler |
| `source` | Source platform name |
| `category` | News category |

Example:

```csv
id,split,label,title,content,spoiler,source,category
cnspoil_train_00001,train,visual-dependent,神秘嘉宾终于现身……,正文内容……,周杰伦,Toutiao,Entertainment
```

The example is synthetic and is provided only to illustrate the schema.

## Removed or Unreleased Fields

### Article URL

The public CSV does not contain a `url` column.

URLs were removed because external news pages can be modified, deleted, redirected, or restricted after collection. Omitting them also reduces dependence on unstable third-party pages and avoids redistributing source metadata that is not required by the released text-generation benchmark.

### Cover Images and Image Paths

The public CSV does not contain `images` or `image_path` columns, and the image files are not included.

The original cover images belong to third-party news publishers or content creators. The project does not redistribute these files without explicit permission. As a result, the public textual release alone cannot reproduce the image-dependent experiments reported in the paper.

Authorized users may create a private local manifest:

```csv
id,image_path
cnspoil_train_00001,/absolute/path/to/image_00001.jpg
```

The training and evaluation scripts join this manifest with the public CSV by `id`.

### Missing or Empty Text

Rows required for model training should contain non-empty values for `id`, `split`, `title`, `content`, and `spoiler`.

A missing article body is not replaced with fabricated text. Such rows should be excluded from training and evaluation because the paper defines the textual input as the concatenation of the headline and article body.

Missing optional descriptive fields such as `source` or `category` may be retained as empty values because they are not model inputs. Their absence should be interpreted as unavailable metadata, not as a new category.

## Label and Visual Dependency

The benchmark distinguishes between text-sufficient and visual-dependent samples. The released `label` field preserves the annotation encoding used in the dataset files.

- **Text-sufficient:** the spoiler can be inferred from the article text.
- **Visual-dependent:** the spoiler requires information from the associated cover image and cannot be reliably recovered from the text alone.

Users should not infer visual dependency from missing images or empty fields. The label is an annotation, while image omission in the public repository is solely a copyright-related release restriction.

## Data Loading Rules

Recommended validation rules:

1. require the exact public columns;
2. require a unique, non-empty `id`;
3. verify that `split` matches the file;
4. reject rows missing `title`, `content`, or `spoiler`;
5. join authorized image paths by `id`;
6. fail explicitly when a multimodal run lacks an image instead of using an artificial placeholder.

## Annotation Pipeline

The annotation workflow consists of:

1. candidate article collection and filtering;
2. candidate spoiler generation;
3. human verification and refinement;
4. visual-dependency verification.

All released spoilers were manually reviewed.

## Research Use

The dataset is released for academic research use only. Users are responsible for complying with applicable copyright rules and source-platform terms.
