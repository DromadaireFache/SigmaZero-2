import os
import struct
import numpy as np
import torch
import torch.nn as nn
import chess
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys
import sigma_zero
import nn_engine
from datasets import load_dataset
import time
from numba import njit


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
        input_vector = fen_to_input(board.fen())
        with torch.no_grad():
            return self.forward(input_vector.to(device)).item()


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


class TrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, max_samples=None, seed=0):
        self.max_samples = max_samples
        self.seed = seed
        self.dataset = load_dataset(
            "Lichess/chess-position-evaluations", split="train", streaming=True, token=os.getenv("HF_TOKEN", None)
        )

    def __len__(self):
        return self.max_samples if self.max_samples is not None else 342059879

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        ds = self.dataset
        ds = ds.shard(num_shards=num_workers, index=worker_id)
        ds = ds.shuffle(seed=self.seed + worker_id)

        limit = None
        if self.max_samples is not None:
            base = self.max_samples // num_workers
            rem = self.max_samples % num_workers
            limit = base + (1 if worker_id < rem else 0)

        produced = 0
        for item in ds:
            cp = item["cp"]
            if cp is None:
                continue
            fen = item["fen"]
            y = float(cp) / 100.0
            yield fen_to_input(fen), torch.tensor(y, dtype=torch.float32)

            produced += 1
            if limit is not None and produced >= limit:
                break


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, max_samples: int = None):
        self.positions = []
        self.evaluations = []
        with open("data/chessData.csv", "r") as f:
            for line in tqdm(f, desc="Loading validation dataset"):
                fen, eval_str = line.strip().split(",")

                try:
                    evaluation = float(eval_str) / 100.0  # Convert centipawns to pawns
                except ValueError:
                    continue  # Skip lines with invalid evaluation values

                self.positions.append(fen_to_input(fen))
                self.evaluations.append(evaluation)
                if max_samples and len(self.positions) >= max_samples:
                    break

        print(f"Loaded {len(self.positions)} positions from data/chessData.csv")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        input_vector = self.positions[idx]
        evaluation = self.evaluations[idx]
        return input_vector, evaluation


def get_baseline_loss() -> float:
    sigma_zero_loss = sigma_zero.command("baseline_loss", JSON=False)
    return float(sigma_zero_loss.strip())


def training_loop(
    model: NNUE, train_set: TrainDataset, val_set: ValDataset, epochs: int = 10, batch_size: int = 100000
):
    criterion = SigmoidScaledMSELoss(k=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    train_losses = []
    val_losses = []

    # Get a baseline for loss using the current static evaluation of SigmaZero engine
    baseline_loss = get_baseline_loss()
    print(f"Baseline validation loss using SigmaZero static evaluation: {baseline_loss:.4f}")

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
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(inputs.to(device))
                loss = criterion(outputs.squeeze(), targets.float().to(device))
                val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"nnue.pth")

    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.axhline(y=baseline_loss, color="r", linestyle="--", label="SigmaZero Baseline Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def load_model(model_path: str) -> NNUE:
    model = NNUE().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def play(board: chess.Board, millis: int, model: NNUE) -> dict:
    """Run a simple minimax search with alpha-beta pruning."""
    engine = _NNUEEngine(board, model, millis)
    return engine.search()


class _NNUEEngine(nn_engine._SearchEngine):
    def __init__(self, board: chess.Board, model: NNUE, millis: int):
        super().__init__(board, model, millis, neg_inf=-10000.0, pos_inf=10000.0, draw_score=0.0)

    def _evaluate(self, board: chess.Board) -> float:
        return self.evaluator.evaluate(board)


if __name__ == "__main__":
    print(f"Using device: {device}")

    commands = ["train", "eval"]
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
            nnue.load_state_dict(torch.load("nnue.pth"))
            print("Loaded existing model from nnue.pth")
        print("NNUE parameters:", sum(p.numel() for p in nnue.parameters()))

        train_set = TrainDataset(max_samples=args.max_samples)
        val_set = ValDataset(max_samples=args.max_samples // 4 if args.max_samples else None)
        training_loop(nnue, train_set, val_set, epochs=args.epochs, batch_size=args.batch_size)

    elif command == "eval":
        parser = argparse.ArgumentParser(description="Evaluate the NNUE model on a chess position")
        parser.add_argument("fen", type=str, help="FEN string of the chess position")
        args = parser.parse_args(sys.argv[2:])

        nnue = NNUE().to(device)
        if os.path.exists("nnue.pth"):
            nnue.load_state_dict(torch.load("nnue.pth"))
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

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
