#!/usr/bin/env python
"""CLI wrapper for evaluation."""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CN-Spoil model")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to YAML config")
    args = parser.parse_args()

    from cn_spoil.config import load_config
    from cn_spoil.evaluate import evaluate

    metrics = evaluate(load_config(args.config))
    print(metrics)
