import os
import struct
import typing
import numpy as np
import torch
import torch.nn as nn
import chess
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys
from datasets import load_dataset
from numba import njit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sigma_zero
from scripts import nn_engine
from scripts.optimizers import best_consts

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


device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


class SigmoidScaledMSELoss(nn.Module):
    def __init__(self, k: float):
        super(SigmoidScaledMSELoss, self).__init__()
        self.k = k

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        scaled_predictions = torch.sigmoid(predictions / self.k)
        scaled_targets = torch.sigmoid(targets / self.k)
        return torch.mean((scaled_predictions - scaled_targets) ** 2)


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

    def evaluate(self, board: chess.Board):
        self.eval()
        input_vector = fen_to_input(board.fen())
        with torch.no_grad():
            return self.forward(input_vector).item()


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


def fen_to_input(fen: str) -> torch.Tensor:
    fen_bytes = np.frombuffer(fen.encode("ascii"), dtype=np.uint8)
    return torch.from_numpy(_fen_bytes_to_array(fen_bytes))


class HFDataset(torch.utils.data.IterableDataset):
    def __init__(self, split: str, max_samples=None, seed=0, val_fraction: float = 0.2):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.val_fraction = val_fraction
        self.dataset = load_dataset(
            "Lichess/chess-position-evaluations", split="train", streaming=True, token=os.getenv("HF_TOKEN", None)
        )

    def _split_limit(self):
        if self.max_samples is None:
            return None
        val_size = int(self.max_samples * self.val_fraction)
        train_size = self.max_samples - val_size
        return train_size if self.split == "train" else val_size

    def __len__(self):
        total = 342059879
        if self.max_samples is not None:
            split_limit = self._split_limit()
            return split_limit if split_limit is not None else total
        if self.split == "train":
            return int(total * (1.0 - self.val_fraction))
        return int(total * self.val_fraction)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        ds = self.dataset.shuffle(seed=self.seed, buffer_size=100_000)
        ds = ds.shard(num_shards=num_workers, index=worker_id)

        limit = None
        split_limit = self._split_limit()
        if split_limit is not None:
            base = split_limit // num_workers
            rem = split_limit % num_workers
            limit = base + (1 if worker_id < rem else 0)

        produced = 0
        val_every = max(int(round(1.0 / self.val_fraction)), 1)
        for idx, item in enumerate(ds):
            is_val = (idx % val_every) == 0
            if (self.split == "val") != is_val:
                continue
            cp = item["cp"]
            if cp is None:
                continue
            fen = item["fen"]
            y = float(cp) / 100.0
            yield fen_to_input(fen), torch.tensor(y, dtype=torch.float32)

            produced += 1
            if limit is not None and produced >= limit:
                break


def training_loop(
    model: NNUE, train_set: HFDataset, val_set: HFDataset, epochs: int = 10, batch_size: int = 100000
):
    criterion = SigmoidScaledMSELoss(k=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    train_losses = []
    val_losses = []

    workers = 2  # Maximum that streaming dataset can handle
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        seen = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.squeeze(), targets.float().to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            seen += inputs.size(0)
        avg_loss = total_loss / seen
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_seen = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(inputs.to(device))
                loss = criterion(outputs.squeeze(), targets.float().to(device))
                val_loss += loss.item() * inputs.size(0)
            val_seen += inputs.size(0)
        avg_val_loss = val_loss / max(val_seen, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"nnue.pth")

    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def load_model(model_path: str) -> NNUE:
    model = NNUE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


class NNUEEngine(nn_engine.SearchEngine):
    def __init__(self, board: chess.Board, model: NNUE, millis: int):
        super().__init__(board, model, millis, neg_inf=-10000.0, pos_inf=10000.0, draw_score=0.0)

    def evaluate(self, board: chess.Board) -> float:
        return self.evaluator.evaluate(board)


def play(board: chess.Board, millis: int, model: NNUE) -> dict:
    """Run a simple minimax search with alpha-beta pruning."""
    engine = NNUEEngine(board, model, millis)
    return engine.search()


class ClassicEngine(nn_engine.SearchEngine):
    PIECE_VALUES = {
        chess.Piece.from_symbol("P"): [x + best_consts["PAWN_VALUE"] for x in best_consts["PS_WHITE_PAWN"]],
        chess.Piece.from_symbol("p"): [x - best_consts["PAWN_VALUE"] for x in best_consts["PS_BLACK_PAWN"]],
        chess.Piece.from_symbol("N"): [x + best_consts["KNIGHT_VALUE"] for x in best_consts["PS_WHITE_KNIGHT"]],
        chess.Piece.from_symbol("n"): [x - best_consts["KNIGHT_VALUE"] for x in best_consts["PS_BLACK_KNIGHT"]],
        chess.Piece.from_symbol("B"): [x + best_consts["BISHOP_VALUE"] for x in best_consts["PS_WHITE_BISHOP"]],
        chess.Piece.from_symbol("b"): [x - best_consts["BISHOP_VALUE"] for x in best_consts["PS_BLACK_BISHOP"]],
        chess.Piece.from_symbol("R"): [x + best_consts["ROOK_VALUE"] for x in best_consts["PS_WHITE_ROOK"]],
        chess.Piece.from_symbol("r"): [x - best_consts["ROOK_VALUE"] for x in best_consts["PS_BLACK_ROOK"]],
        chess.Piece.from_symbol("Q"): [x + best_consts["QUEEN_VALUE"] for x in best_consts["PS_WHITE_QUEEN"]],
        chess.Piece.from_symbol("q"): [x - best_consts["QUEEN_VALUE"] for x in best_consts["PS_BLACK_QUEEN"]],
        chess.Piece.from_symbol("K"): [x + best_consts["KING_VALUE"] for x in best_consts["PS_WHITE_KING"]],
        chess.Piece.from_symbol("k"): [x - best_consts["KING_VALUE"] for x in best_consts["PS_BLACK_KING"]],
    }

    def __init__(self, board: chess.Board, millis: int):
        super().__init__(board, None, millis, neg_inf=-10000.0, pos_inf=10000.0, draw_score=0.0)

    def evaluate(self, board: chess.Board) -> float:
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                score += ClassicEngine.PIECE_VALUES[piece][square]
        return score / 100.0  # Convert to pawns


def python_play(board: chess.Board, millis: int) -> dict:
    """Run a simple minimax search with alpha-beta pruning."""
    engine = ClassicEngine(board, millis)
    return engine.search()


def quantize(model: NNUE):
    state = model.state_dict()
    QA, QB = 255, 64

    fc1_w = state["fc1.weight"].numpy() * QA
    fc2_w = state["fc2.weight"].numpy() * QB
    fc3_w = state["fc3.weight"].numpy() * QB
    fc1_b = state["fc1.bias"].numpy() * QA
    fc2_b = state["fc2.bias"].numpy() * QB
    fc3_b = state["fc3.bias"].numpy() * QB

    print(f"L0 weights range: [{fc1_w.min():.1f}, {fc1_w.max():.1f}] (int16 range: ±32767)")
    print(f"L1 weights range: [{fc2_w.min():.1f}, {fc2_w.max():.1f}] (int8 range: ±127)")
    print(f"L2 weights range: [{fc3_w.min():.1f}, {fc3_w.max():.1f}] (int8 range: ±127)")

    def write_array(f: typing.TextIO, name: str, array: np.ndarray, dtype: str, max_val: int):
        if len(array.shape) == 1:
            f.write(f"const {dtype} {name}[{len(array)}] = {{")
        else:
            f.write(f"const {dtype} {name}[{len(array)}][{array.shape[1]}] = {{\n")
        for i in range(len(array)):
            if len(array.shape) == 1:
                w = int(np.clip(array[i], -max_val, max_val))
                f.write(f"{w}, " if i < len(array) - 1 else f"{w}")
            else:
                f.write("    {")
                for j in range(array.shape[1]):
                    w = int(np.clip(array[i, j], -max_val, max_val))
                    f.write(f"{w}, " if j < array.shape[1] - 1 else f"{w}")
                f.write("},\n")
        f.write("};\n\n")

    with open("src/nnue/params.c", "w") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"const int QA = {QA}, QB = {QB};\n\n")
        write_array(f, "fc1_weights", fc1_w, "int16_t", 32767)
        write_array(f, "fc2_weights", fc2_w, "int8_t", 127)
        write_array(f, "fc3_weights", fc3_w, "int8_t", 127)
        write_array(f, "fc1_biases", fc1_b, "int16_t", 32767)
        write_array(f, "fc2_biases", fc2_b, "int8_t", 127)
        write_array(f, "fc3_biases", fc3_b, "int8_t", 127)


if __name__ == "__main__":
    print(f"Using device: {device}")

    commands = ["train", "eval", "quantize"]
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: {sys.argv[0]} <command> [options]")
        print(f"Commands: {', '.join(commands)}")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        parser = argparse.ArgumentParser(description="Train the NNUE model")
        parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples")
        parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
        args = parser.parse_args(sys.argv[2:])

        nnue = NNUE().to(device)
        if os.path.exists("nnue.pth"):
            nnue.load_state_dict(torch.load("nnue.pth", map_location=device))
            print("Loaded existing model from nnue.pth")
        print("NNUE parameters:", sum(p.numel() for p in nnue.parameters()))

        train_set = HFDataset(split="train", max_samples=args.max_samples)
        val_set = HFDataset(split="val", max_samples=args.max_samples)
        training_loop(nnue, train_set, val_set, epochs=args.epochs, batch_size=args.batch_size)

    elif command == "eval":
        parser = argparse.ArgumentParser(description="Evaluate the NNUE model on a chess position")
        parser.add_argument("fen", type=str, help="FEN string of the chess position")
        args = parser.parse_args(sys.argv[2:])

        nnue = NNUE()
        if os.path.exists("nnue.pth"):
            nnue.load_state_dict(torch.load("nnue.pth", map_location=device))
            print("Loaded existing model from nnue.pth")
        else:
            print("No trained model found. Please train the model first.")
            sys.exit(1)

        board = chess.Board(args.fen)
        evaluation = nnue.evaluate(board)
        print(f"NNUE Evaluation for FEN '{args.fen}': {evaluation:.4f} pawns")

        # Evaluate moves at depth 1 and print the 3 best moves according to the NNUE evaluation
        best_moves = []
        for move in board.legal_moves:
            board.push(move)
            move_eval = nnue.evaluate(board)
            best_moves.append((move, move_eval))
            board.pop()
        best_moves.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        print("Top 3 moves according to NNUE evaluation:")
        for move, eval in best_moves[:3]:
            print(f"  Move: {move}, Evaluation: {eval:.4f} pawns")

    elif command == "quantize":
        model = NNUE()
        if os.path.exists("nnue.pth"):
            model.load_state_dict(torch.load("nnue.pth", map_location=device))
            print("Loaded existing model from nnue.pth")
        else:
            print("No trained model found. Please train the model first.")
            sys.exit(1)

        quantize(model)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
