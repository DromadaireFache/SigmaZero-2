import os

import chess
import numpy as np
import torch
import torch.nn as nn
from numba import njit
from abc import ABC

from .chessnn import ChessNN

# ASCII lookup table for piece-plane mapping: black first, then white.
PIECE_TO_PLANE = np.full(128, -1, dtype=np.int16)
PIECE_TO_PLANE[ord("p")] = 0
PIECE_TO_PLANE[ord("P")] = 1
PIECE_TO_PLANE[ord("n")] = 2
PIECE_TO_PLANE[ord("N")] = 3
PIECE_TO_PLANE[ord("b")] = 4
PIECE_TO_PLANE[ord("B")] = 5
PIECE_TO_PLANE[ord("r")] = 6
PIECE_TO_PLANE[ord("R")] = 7
PIECE_TO_PLANE[ord("q")] = 8
PIECE_TO_PLANE[ord("Q")] = 9
PIECE_TO_PLANE[ord("k")] = 10
PIECE_TO_PLANE[ord("K")] = 11


@njit(cache=True)
def _fen_bytes_to_array(fen_bytes: np.ndarray) -> np.ndarray:
    # Parse ASCII FEN bytes directly in nopython mode.
    input_vector = np.zeros(769, dtype=np.float32)
    row, col = 7, 0
    phase = 0  # 0: board, 1: side-to-move, 2: remaining fields

    for i in range(fen_bytes.shape[0]):
        c = fen_bytes[i]

        if phase == 0:
            if c == 32:  # ' '
                phase = 1
            elif c == 47:  # '/'
                row -= 1
                col = 0
            elif 49 <= c <= 56:  # '1'..'8'
                col += c - 48
            else:
                index = PIECE_TO_PLANE[c]
                if index >= 0:
                    input_vector[(index << 6) + (row << 3) + col] = 1.0
                    col += 1

        elif phase == 1:
            input_vector[768] = 1.0 if c == 119 else 0.0  # 'w'
            phase = 2

        elif c == 32:
            break

    return input_vector


class ClassicArch(ChessNN, ABC):
    def __init__(self, model_path: str):
        super(ClassicArch, self).__init__(model_path)

    def evaluate(self, board: chess.Board) -> float:
        self.eval()
        input_vector = self.fen_to_input(board.fen())
        with torch.no_grad():
            return self.forward(input_vector).item()

    def fen_to_input(self, fen: str) -> torch.Tensor:
        fen_bytes = np.frombuffer(fen.encode("ascii"), dtype=np.uint8)
        return torch.from_numpy(_fen_bytes_to_array(fen_bytes))


class Arch1(ClassicArch):
    def __init__(self, dropout_p: float = 0.10):
        super(Arch1, self).__init__("nnue/models/arch1.pth")
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


class Arch2(ClassicArch):
    def __init__(self, dropout_p: float = 0.10):
        super(Arch2, self).__init__("nnue/models/arch2.pth")
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
    
    
class Arch3(ClassicArch):
    def __init__(self, dropout_p: float = 0.10):
        super(Arch3, self).__init__("nnue/models/arch3.pth")
        self.fc1 = nn.Linear(769, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 128)
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
    
    
class Tiny(ClassicArch):
    def __init__(self, dropout_p: float = 0.10):
        super(Tiny, self).__init__("nnue/models/tiny.pth")
        self.fc1 = nn.Linear(769, 32)
        self.fc2 = nn.Linear(32, 64)
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


if __name__ == "__main__":
    # Benchmark the fen_to_input function.
    import time
    import sys
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from scripts.chessdata import Dataloader
    
    # Warm up the JIT compiler.
    for fen in Dataloader(100).fens():
        _ = _fen_bytes_to_array(np.frombuffer(fen.encode("ascii"), dtype=np.uint8))
    
    fens = Dataloader(1e5).fens()
    start_time = time.perf_counter()
    
    for fen in fens:
        _ = _fen_bytes_to_array(np.frombuffer(fen.encode("ascii"), dtype=np.uint8))
    
    end_time = time.perf_counter()
    print(f"Processed 100k FENs in {end_time - start_time:.2f} seconds")