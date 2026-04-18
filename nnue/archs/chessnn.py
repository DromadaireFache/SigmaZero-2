import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import chess


class ChessNN(nn.Module, ABC):
    def __init__(self, model_path: str):
        super(ChessNN, self).__init__()
        self.model_path = model_path

    def load_model(self, device: torch.device = torch.device("cpu"), allow_missing: bool = False):
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.load_state_dict(state_dict)
            self.eval()
            if allow_missing:
                print(f"Loading model from checkpoint: {self.model_path}")
        else:
            if allow_missing:
                print(f"No model file found at {self.model_path}. Starting with a new model.")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
    def save_model(self):
        torch.save(self.state_dict(), self.model_path)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        pass

    @abstractmethod
    def fen_to_input(self, fen: str) -> torch.Tensor:
        pass
