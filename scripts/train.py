#!/usr/bin/env python
"""CLI wrapper for training."""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CN-Spoil model")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    args = parser.parse_args()

    from cn_spoil.config import load_config
    from cn_spoil.train import train

    checkpoint = train(load_config(args.config))
    print(f"Saved checkpoint: {checkpoint}")
