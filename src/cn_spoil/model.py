"""Placeholder CN-Spoil model."""

import torch
from torch import nn


class GatedDualEncoder(nn.Module):
    """Lightweight gated dual-encoder placeholder."""

    def __init__(self, text_dim: int = 128, visual_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        text_h = self.text_proj(text_features)
        visual_h = self.visual_proj(visual_features)
        fused = torch.cat([text_h, visual_h], dim=-1)
        gate = self.gate(fused)
        mixed = gate * text_h + (1.0 - gate) * visual_h
        return self.decoder(mixed).squeeze(-1)
