# NNUE Architecture comparison

## Overview

Training results for different architectures over 3 epochs and 250M samples.

|                                         | Parameters | Train 10 | Val 10  | Train 20 | Val 20  | Train 30 | Val 30  |
|-----------------------------------------|------------|----------|---------|----------|---------|----------|---------|
| [Arch1](#arch1)                         | 213633     | 0.0170   | 0.0170  | 0.0167   | 0.0167  | 0.0167   | 0.0166  |
| [Arch1-Small](#arch1-small)             | 102721     |
| [Arch1-Large](#arch1-large)             | 460033     |
| [Arch1-Tiny](#arch1-tiny)               | 12897      |
| [Arch2](#arch2)                         | 1083905    |
| [ResNet](#resnet)                       | 3558913    | 0.0145   | 0.0160  | 0.0144   | 0.0159  | 0.0130   | 0.0140  |

## Arch1

```python
class NNUE(nn.Module):
    def __init__(self, dropout_p: float = 0.10):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(769, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.clamp(torch.relu(self.fc1(x)), 0, 1)
        x = self.drop1(x)
        x = torch.clamp(torch.relu(self.fc2(x)), 0, 1)
        x = self.drop2(x)
        x = self.fc3(x)
        return x
```

## Arch1-Small

```python
class NNUE(nn.Module):
    def __init__(self, dropout_p: float = 0.10):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(769, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        ... # same as Arch1
```

## Arch1-Large

```python
class NNUE(nn.Module):
    def __init__(self, dropout_p: float = 0.10):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(769, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)
    
    def forward(self, x):
        ... # same as Arch1
```

## Arch1-Tiny

```python
class NNUE(nn.Module):
    def __init__(self, dropout_p: float = 0.10):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(769, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        ... # same as Arch1
```

## Arch2

```python
class NNUE(nn.Module):
    def __init__(self, dropout_p: float = 0.10):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(769, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)
        self.drop3 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.clamp(torch.relu(self.fc1(x)), 0, 1)
        x = self.drop1(x)
        x = torch.clamp(torch.relu(self.fc2(x)), 0, 1)
        x = self.drop2(x)
        x = torch.clamp(torch.relu(self.fc3(x)), 0, 1)
        x = self.drop3(x)
        x = self.fc4(x)
        return x
```

## ResNet

```python
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = torch.clamp(self.fc1(self.norm1(x)), 0, 1)
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
```