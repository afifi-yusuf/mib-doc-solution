from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layers(x) + self.skip(x))


class PacketCNN(nn.Module):
    """Small custom vision-only page encoder with ordered-page attention pooling."""
    def __init__(self, task_sizes: dict[str, int], width: int = 32, max_pages: int = 6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 5, 2, 2), nn.BatchNorm2d(width), nn.ReLU())
        self.encoder = nn.Sequential(
            ConvBlock(width, width), ConvBlock(width, width * 2, 2),
            ConvBlock(width * 2, width * 4, 2), ConvBlock(width * 4, width * 6, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        dim = width * 6
        self.page_position = nn.Parameter(torch.zeros(1, max_pages, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.heads = nn.ModuleDict({name: nn.Linear(dim, size) for name, size in task_sizes.items()})

    def forward(self, pages: torch.Tensor, page_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, count, channels, height, width = pages.shape
        features = self.encoder(self.stem(pages.reshape(batch * count, channels, height, width)))
        features = self.pool(features).flatten(1).reshape(batch, count, -1)
        features = features + self.page_position[:, :count]
        attended, _ = self.attention(features, features, features, key_padding_mask=~page_mask)
        packet = (attended * page_mask.unsqueeze(-1)).sum(1) / page_mask.sum(1, keepdim=True).clamp_min(1)
        return {name: head(packet) for name, head in self.heads.items()}

