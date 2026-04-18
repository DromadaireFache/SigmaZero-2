import glob
import os
import shutil
import typing
import time
import numpy as np
import torch
import torch.nn as nn
import chess
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnue import archs
from nnue.archs.chessnn import ChessNN
from nnue.data import HFDataset, ensure_local_dataset, DEFAULT_HF_DATASET_DIR
from scripts import nn_engine
from scripts.engines import Engine
from scripts.optimizers import best_consts


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


def training_loop(
    model: ChessNN,
    train_set: HFDataset,
    val_set: HFDataset,
    epochs: int = 10,
    batch_size: int = 100000,
    workers: int = 2,
):
    criterion = SigmoidScaledMSELoss(k=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    train_losses = []
    val_losses = []

    loader_kwargs = {
        "batch_size": None,
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **loader_kwargs)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        seen = 0
        train_iter = iter(train_loader)
        train_pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}/{epochs} - Training")
        step = 0
        ds_fetch_total = 0.0
        ds_norm_total = 0.0
        transfer_total = 0.0
        step_total = 0.0
        while True:
            wait_start = time.perf_counter()
            try:
                inputs, targets, ds_fetch_times, ds_norm_times = next(train_iter)
            except StopIteration:
                break
            wait_time = time.perf_counter() - wait_start
            raw_ds_fetch = float(ds_fetch_times)
            raw_ds_norm = float(ds_norm_times)
            raw_ds_total = raw_ds_fetch + raw_ds_norm
            if raw_ds_total > 0.0:
                ds_fetch_total += wait_time * (raw_ds_fetch / raw_ds_total)
                ds_norm_total += wait_time * (raw_ds_norm / raw_ds_total)
            else:
                ds_fetch_total += wait_time

            batch_size_actual = inputs.size(0)
            for start in range(0, batch_size_actual, batch_size):
                end = min(start + batch_size, batch_size_actual)
                input_chunk = inputs[start:end]
                target_chunk = targets[start:end]

                transfer_start = time.perf_counter()
                input_chunk = input_chunk.to(device)
                target_chunk = target_chunk.float().to(device)
                transfer_time = time.perf_counter() - transfer_start

                step_start = time.perf_counter()
                optimizer.zero_grad()
                outputs = model(input_chunk)
                loss = criterion(outputs.squeeze(), target_chunk)
                loss.backward()
                optimizer.step()
                step_time = time.perf_counter() - step_start

                total_loss += loss.item() * input_chunk.size(0)
                seen += input_chunk.size(0)
                step += 1
                transfer_total += transfer_time
                step_total += step_time
                train_pbar.update(input_chunk.size(0))
        train_pbar.close()
        avg_loss = total_loss / seen
        train_losses.append(avg_loss)
        print(
            "Epoch "
            f"{epoch+1}/{epochs} - Training Loss: {avg_loss:.4f} "
            f"(ds_fetch {ds_fetch_total:.2f}s, "
            f"ds_normalize {ds_norm_total:.2f}s, "
            f"to_device {transfer_total:.2f}s, "
            f"step {step_total:.2f}s)"
        )

        # Validation
        model.eval()
        val_loss = 0
        val_seen = 0
        val_steps = 0
        val_ds_fetch_total = 0.0
        val_ds_norm_total = 0.0
        val_transfer_total = 0.0
        with torch.no_grad():
            val_iter = iter(val_loader)
            val_pbar = tqdm(total=len(val_loader.dataset), desc=f"Epoch {epoch+1}/{epochs} - Validation")
            while True:
                wait_start = time.perf_counter()
                try:
                    inputs, targets, ds_fetch_times, ds_norm_times = next(val_iter)
                except StopIteration:
                    break
                wait_time = time.perf_counter() - wait_start
                val_steps += 1
                raw_ds_fetch = float(ds_fetch_times)
                raw_ds_norm = float(ds_norm_times)
                raw_ds_total = raw_ds_fetch + raw_ds_norm
                if raw_ds_total > 0.0:
                    val_ds_fetch_total += wait_time * (raw_ds_fetch / raw_ds_total)
                    val_ds_norm_total += wait_time * (raw_ds_norm / raw_ds_total)
                else:
                    val_ds_fetch_total += wait_time

                batch_size_actual = inputs.size(0)
                for start in range(0, batch_size_actual, batch_size):
                    end = min(start + batch_size, batch_size_actual)
                    input_chunk = inputs[start:end]
                    target_chunk = targets[start:end]

                    transfer_start = time.perf_counter()
                    input_chunk = input_chunk.to(device)
                    target_chunk = target_chunk.float().to(device)
                    val_transfer_total += time.perf_counter() - transfer_start

                    outputs = model(input_chunk)
                    loss = criterion(outputs.squeeze(), target_chunk)
                    val_loss += loss.item() * input_chunk.size(0)
                    val_seen += input_chunk.size(0)
                    val_pbar.update(input_chunk.size(0))
            val_pbar.close()
        avg_val_loss = val_loss / max(val_seen, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        print(
            f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f} "
            f"(ds_fetch {val_ds_fetch_total:.2f}s, "
            f"ds_normalize {val_ds_norm_total:.2f}s, "
            f"to_device {val_transfer_total:.2f}s)"
        )

        # Save model checkpoint
        model.save_model()

    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
    plt.savefig("training_plot.png")


class ChessNNSearch(nn_engine.SearchEngine):
    def __init__(self, board: chess.Board, model: ChessNN, millis: int):
        super().__init__(board, model, millis, neg_inf=-10000.0, pos_inf=10000.0, draw_score=0.0)

    def evaluate(self, board: chess.Board) -> float:
        return self.evaluator.evaluate(board)


class ChessNNEngine(Engine):
    def __init__(self, model: ChessNN):
        self.model = model

    def play(self, board: chess.Board, millis: int) -> dict:
        """Run a simple minimax search with alpha-beta pruning."""
        engine = ChessNNSearch(board, self.model, millis)
        return engine.search()
        # return self.play_depth_1(board, millis)
    
    def play_depth_1(self, board: chess.Board, millis: int) -> dict:
        """Evaluate all legal moves at depth 1 and return the best move."""
        best_move = None
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                move_eval = 10000.0 if board.turn == chess.BLACK else -10000.0
            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
                move_eval = 0.0
            else:
                move_eval = self.model.evaluate(board)
            board.pop()
            if (board.turn == chess.WHITE and move_eval > best_eval) or (board.turn == chess.BLACK and move_eval < best_eval):
                best_eval = move_eval
                best_move = move
        return {"move": best_move.uci(), "eval": round(best_eval, 2)}

    def version(self) -> str:
        return f"ChessNN-{self.model.__class__.__name__}"


class PySearch(nn_engine.SearchEngine):
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
                score += PySearch.PIECE_VALUES[piece][square]
        return score / 100.0  # Convert to pawns


class PyEngine(Engine):
    def play(self, board: chess.Board, millis: int) -> dict:
        engine = PySearch(board, millis)
        return engine.search()

    def version(self) -> str:
        return "ClassicEngine"


if __name__ == "__main__":
    # Use 'spawn' start method for multiprocessing to avoid CUDA re-initialization in forked subprocesses.
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    print(f"Using device: {device}")

    commands = ["train", "eval", "quantize"]
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: {sys.argv[0]} <command> [options]")
        print(f"Commands: {', '.join(commands)}")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        parser = argparse.ArgumentParser(description="Train a ChessNN model")
        parser.add_argument("chess_nn", type=str, help="ChessNN architecture to use (e.g. 'arch1')")
        parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples")
        parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
        parser.add_argument(
            "--dataset-dir",
            type=str,
            default=DEFAULT_HF_DATASET_DIR,
            help="Local dataset path used for downloaded HF data",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=None,
            help="DataLoader workers (default: 0 for local Arrow batches)",
        )
        args = parser.parse_args(sys.argv[2:])

        ensure_local_dataset(args.dataset_dir)

        chess_nn = archs.get_arch(args.chess_nn)
        chess_nn.load_model(allow_missing=True)
        chess_nn.to(device)
        print("Model parameters:", sum(p.numel() for p in chess_nn.parameters()))

        if args.workers is not None:
            workers = args.workers
        else:
            workers = os.cpu_count()
        print(f"DataLoader workers: {workers}")

        train_set = HFDataset(
            chess_nn,
            split="train",
            max_samples=args.max_samples,
            dataset_dir=args.dataset_dir,
        )
        val_set = HFDataset(
            chess_nn,
            split="val",
            max_samples=args.max_samples,
            dataset_dir=args.dataset_dir,
        )
        training_loop(chess_nn, train_set, val_set, epochs=args.epochs, batch_size=args.batch_size, workers=workers)

    elif command == "eval":
        parser = argparse.ArgumentParser(description="Evaluate a ChessNN model on a chess position")
        parser.add_argument("chess_nn", type=str, help="ChessNN architecture to use (e.g. 'arch1')")
        parser.add_argument("fen", type=str, help="FEN string of the chess position")
        args = parser.parse_args(sys.argv[2:])

        chess_nn = archs.get_arch(args.chess_nn)
        chess_nn.load_model(allow_missing=False)

        board = chess.Board(args.fen)
        evaluation = chess_nn.evaluate(board)
        print(f"{args.chess_nn} Evaluation for FEN '{args.fen}': {evaluation:.4f} pawns")

        # Evaluate moves at depth 1 and print the 3 best moves according to the ChessNN evaluation
        best_moves = []
        for move in board.legal_moves:
            board.push(move)
            move_eval = chess_nn.evaluate(board)
            best_moves.append((move, move_eval))
            board.pop()
        best_moves.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        print(f"Top 3 moves according to {args.chess_nn} evaluation:")
        for move, eval in best_moves[:3]:
            print(f"  Move: {move}, Evaluation: {eval:.4f} pawns")

    # elif command == "quantize":
    #     model = Arch1()
    #     if os.path.exists("nnue.pth"):
    #         model.load_state_dict(torch.load("nnue.pth", map_location=device))
    #         print("Loaded existing model from nnue.pth")
    #     else:
    #         print("No trained model found. Please train the model first.")
    #         sys.exit(1)

    #     quantize(model)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
