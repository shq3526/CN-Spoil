#!/usr/bin/env python
"""CLI wrapper for inference."""

import json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CN-Spoil inference")
    parser.add_argument("--config", default="configs/infer.yaml", help="Path to YAML config")
    parser.add_argument(
        "--input-json",
        default='{"title": "", "article": "", "image_caption": ""}',
        help="JSON object with title/article/image_caption",
    )
    args = parser.parse_args()

    from cn_spoil.config import load_config
    from cn_spoil.inference import predict

    sample = json.loads(args.input_json)
    output = predict(load_config(args.config), sample)
    print(json.dumps(output, ensure_ascii=False))
