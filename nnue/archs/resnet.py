import torch.nn as nn
import torch

from .classic import ClassicArch


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout_p)
        
    def forward(self, x):
        residual = x
        x = torch.clamp(self.fc1(self.norm1(x)), 0, 1)
        x = self.drop(x)
        x = self.fc2(self.norm2(x))
        return x + residual


class ChessResNet(ClassicArch):
    def __init__(self):
        super().__init__("nnue/models/resnet.pth")
        self.input_proj = nn.Linear(769, 512)
        self.blocks = nn.Sequential(*[ResBlock(512) for _ in range(6)])
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.clamp(self.input_proj(x), 0, 1)
        x = self.blocks(x)
        return self.output(x)
