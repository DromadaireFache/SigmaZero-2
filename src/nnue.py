import os

import torch
import torch.nn as nn
import chess
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys
import sigma_zero


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
        self.fc1 = nn.Linear(769, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
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
        input_vector = board_to_input(board)
        with torch.no_grad():
            return self.forward(input_vector).item()


def board_to_input(board: chess.Board):
    input_vector = torch.zeros(769, dtype=torch.float32)  # 768 for pieces + 1 for side to move
    input_vector[768] = float(board.turn)  # 1.0 for white to move, 0.0 for black to move
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            color = int(piece.color)
            index = (piece_type - 1) * 2 + color
            input_vector[index * 64 + square] = 1
    return input_vector


def print_board(board: chess.Board):
    print(board.fen(), "as input vector:")
    input_vector = board_to_input(board)
    for i in range(12):
        piece_name = chess.PIECE_NAMES[i // 2 + 1]
        color = "white" if i % 2 == 0 else "black"
        print(f"{color} {piece_name}: {"".join(str(int(x)) for x in input_vector[i*64:(i+1)*64].tolist())}")
    print(f"Side to move: {'white' if input_vector[768] == 1.0 else 'black'}")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, chess_data_path: str, max_samples: int = None):
        self.positions = []
        self.evaluations = []
        with open(chess_data_path, "r") as f:
            for line in tqdm(f, desc="Loading dataset"):
                fen, eval_str = line.strip().split(",")

                try:
                    evaluation = float(eval_str) / 100.0  # Convert centipawns to pawns
                except ValueError:
                    continue  # Skip lines with invalid evaluation values

                self.positions.append(fen)
                self.evaluations.append(evaluation)
                if max_samples and len(self.positions) >= max_samples:
                    break

        print(f"Loaded {len(self.positions)} positions from {chess_data_path}")

    def convert_fens_to_inputs(self):
        self.positions = [
            board_to_input(chess.Board(fen)) for fen in tqdm(self.positions, desc="Converting FENs to input vectors")
        ]
        self.evaluations = torch.tensor(self.evaluations, dtype=torch.float32)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        input_vector = self.positions[idx]
        evaluation = self.evaluations[idx]
        return input_vector, evaluation

    def split(self, train_ratio: float = 0.8):
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        return torch.utils.data.random_split(self, [train_size, val_size])


def get_baseline_loss(val_set: torch.utils.data.Subset):
    with open("data/val_set.csv", "w") as f:
        for fen, e in val_set:
            f.write(f"{fen},{e}\n")
    sigma_zero_loss = sigma_zero.command("baseline_loss", JSON=False)
    os.remove("data/val_set.csv")
    return float(sigma_zero_loss.strip())


def training_loop(model: NNUE, dataset: Dataset, epochs: int = 10, batch_size: int = 100000):
    train_set, val_set = dataset.split()
    criterion = SigmoidScaledMSELoss(k=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    train_losses = []
    val_losses = []

    # Get a baseline for loss using the current static evaluation of SigmaZero engine
    baseline_loss = get_baseline_loss(val_set)
    print(f"Baseline validation loss using SigmaZero static evaluation: {baseline_loss:.4f}")

    dataset.convert_fens_to_inputs()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.float())
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
    model = NNUE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def play(board: chess.Board, milliseconds: int, model: NNUE):
    best_move = None
    best_eval = float("-inf") if board.turn == chess.WHITE else float("inf")
    for move in board.legal_moves:
        board.push(move)
        move_eval = model.evaluate(board)
        board.pop()
        if (board.turn == chess.WHITE and move_eval > best_eval) or (
            board.turn == chess.BLACK and move_eval < best_eval
        ):
            best_eval = move_eval
            best_move = move
    return {"move": best_move.uci() if best_move else None, "eval": round(best_eval, 2)}


if __name__ == "__main__":
    commands = ["train", "eval"]
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: {sys.argv[0]} <command> [options]")
        print(f"Commands: {', '.join(commands)}")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        parser = argparse.ArgumentParser(description="Train the NNUE model")
        parser.add_argument("--data", type=str, default="data/chessData.csv", help="Path to the chess data CSV file")
        parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples")
        parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
        args = parser.parse_args(sys.argv[2:])

        nnue = NNUE()
        if os.path.exists("nnue.pth"):
            nnue.load_state_dict(torch.load("nnue.pth"))
            print("Loaded existing model from nnue.pth")
        print("NNUE parameters:", sum(p.numel() for p in nnue.parameters()))

        dataset = Dataset(args.data, max_samples=args.max_samples)
        training_loop(nnue, dataset, epochs=args.epochs, batch_size=args.batch_size)

    elif command == "eval":
        parser = argparse.ArgumentParser(description="Evaluate the NNUE model on a chess position")
        parser.add_argument("fen", type=str, help="FEN string of the chess position")
        args = parser.parse_args(sys.argv[2:])

        nnue = NNUE()
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
