"""Training entrypoints for CN-Spoil."""

from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .datasets import CNSpoilDataset, collate_batch
from .model import GatedDualEncoder
from .utils import set_seed


def train(config: Dict[str, Any]) -> Path:
    set_seed(int(config.get("seed", 42)))

    train_dataset = CNSpoilDataset(config["data"]["train_path"], max_samples=config["data"].get("max_samples"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 4)),
        shuffle=True,
        collate_fn=collate_batch,
    )

    model = GatedDualEncoder(
        text_dim=int(config["model"].get("text_dim", 128)),
        visual_dim=int(config["model"].get("visual_dim", 128)),
        hidden_dim=int(config["model"].get("hidden_dim", 128)),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-3)))

    for _ in range(int(config["training"].get("epochs", 1))):
        for batch in train_loader:
            bsz = len(batch["title"])
            text_features = torch.randn(bsz, model.text_proj.in_features)
            visual_features = torch.randn(bsz, model.visual_proj.in_features)
            targets = torch.tensor([min(len(s), 1) for s in batch["spoiler"]], dtype=torch.float32)

            logits = model(text_features, visual_features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / "cn_spoil_latest.pt"
    torch.save(model.state_dict(), output_path)
    return output_path
