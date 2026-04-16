# NNUE Architecture comparison

## Overview

Training results for different architectures over 3 epochs and 250M samples.

|                                         | Parameters | Train 1 | Val 1  | Train 2 | Val 2  | Train 3 | Val 3  |
|-----------------------------------------|------------|---------|--------|---------|--------|---------|--------|
| [Arch1](#arch1)                         | 213633     | 0.0201  | 0.0222 | 0.0197  | 0.0220 | 0.0197  | 0.0222 |
| [Arch1-Small](#arch1-small)             | 102721     | 0.0203  | 0.0218 | 0.0199  | 0.0217 | 0.0199  | 0.0218 |
| [Arch1-Large](#arch1-large)             | 460033     | 0.0199  | 0.0224 | 0.0196  | 0.0221 | 0.0196  | 0.0223 |
| [Arch1-Tiny](#arch1-tiny)               | 12897      | 0.0219  | 0.0224 | 0.0217  | 0.0224 | 0.0216  | 0.0224 |
| [Arch1 (lr=1e-4)](#arch1)               | 213633     | 0.0187  | 0.0187 | 0.0175  | 0.0181 | 0.0172  | 0.0179 |
| [Arch1-Small (lr=1e-4)](#arch1-small)   | 102721     | 0.0194  | 0.0190 | 0.0182  | 0.0186 | 0.0180  | 0.0183 |
| [Arch2](#arch2)                         | 1083905    |

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