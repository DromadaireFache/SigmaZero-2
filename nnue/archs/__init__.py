import sys

import torch

from .classic import Arch1, Arch2, Arch3, Tiny
from .resnet import ChessResNet
from .chessnn import ChessNN

architectures: dict[str, type[ChessNN]] = {
    "arch1": Arch1,
    "arch2": Arch2,
    "arch3": Arch3,
    "tiny": Tiny,
    "resnet": ChessResNet,
}

def get_arch(chess_nn_name: str) -> ChessNN:
    if chess_nn_name not in architectures:
        print(f"Unknown ChessNN architecture: {chess_nn_name}")
        print(f"Available architectures: {list(architectures.keys())}")
        sys.exit(1)
    return architectures[chess_nn_name]()


def get_all_archs() -> dict[str, ChessNN]:
    """Returns a dictionary of all available ChessNN architectures with pre-trained parameters."""
    result = {}
    for name, arch_cls in architectures.items():
        try:
            arch = arch_cls()
            arch.load_model()
            result[name] = arch
        except FileNotFoundError:
            print(f"Warning: Pre-trained model for {name} not found. Skipping.")
    return result