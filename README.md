# CN-Spoil

Official repository for CN-Spoil: the first large-scale multimodal Chinese clickbait spoiling dataset and a lightweight Gated Dual-Encoder framework for visual-grounded spoiler generation. Accepted at ECML PKDD 2026 (Applied Data Science Track).

## Repository layout

```text
CN-Spoil/
├── checkpoints/pretrained/      # placeholder for released checkpoints
├── configs/                     # experiment YAML configs
├── data/CN-Spoil/               # placeholder for dataset files
├── scripts/                     # train/eval/infer entry scripts
├── src/cn_spoil/                # package code (dataset/model/train/eval/infer)
└── tests/                       # scaffold tests
```

## Quick start

1. Create an environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Put dataset JSONL files in `data/CN-Spoil/`.
3. Put pretrained weights in `checkpoints/pretrained/cn_spoil_ecml_pkdd_2026.pt`.

## Running experiments

```bash
PYTHONPATH=src python scripts/train.py --config configs/train.yaml
PYTHONPATH=src python scripts/evaluate.py --config configs/eval.yaml
PYTHONPATH=src python scripts/infer.py --config configs/infer.yaml --input-json '{"title":"示例标题","article":"示例正文","image_caption":"示例图像描述"}'
```

## Reproducibility (ECML PKDD 2026)

- Fix random seed in config files (`seed: 42` by default).
- Record the exact commit hash and config used for each run.
- Keep train/dev/test splits unchanged from the official release.
- Save checkpoints and metrics for each run under a timestamped experiment folder.
- Report mean and standard deviation over at least 3 runs for final results.

Dataset release details and final benchmark scripts will be added here after camera-ready artifact packaging.
