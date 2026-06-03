"""Inference entrypoints for CN-Spoil."""

from pathlib import Path
from typing import Any, Dict

import torch

from .model import GatedDualEncoder


def generate_spoiler(title: str, article: str, image_caption: str) -> str:
    del article, image_caption
    return f"[placeholder spoiler] {title[:64]}".strip()


def predict(config: Dict[str, Any], sample: Dict[str, str]) -> Dict[str, str]:
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

    del model
    spoiler = generate_spoiler(
        title=sample.get("title", ""),
        article=sample.get("article", ""),
        image_caption=sample.get("image_caption", ""),
    )
    return {"spoiler": spoiler}
