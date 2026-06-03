"""Evaluation entrypoints for CN-Spoil."""

from pathlib import Path
from typing import Any, Dict

import torch

from .datasets import CNSpoilDataset
from .model import GatedDualEncoder


def evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    dataset = CNSpoilDataset(config["data"]["eval_path"], max_samples=config["data"].get("max_samples"))

    model = GatedDualEncoder(
        text_dim=int(config["model"].get("text_dim", 128)),
        visual_dim=int(config["model"].get("visual_dim", 128)),
        hidden_dim=int(config["model"].get("hidden_dim", 128)),
    )

    checkpoint_path = config["paths"].get("checkpoint_path")
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    avg_spoiler_len = sum(len(item["spoiler"]) for item in dataset) / max(len(dataset), 1)
    return {"samples": float(len(dataset)), "avg_spoiler_len": float(avg_spoiler_len)}
