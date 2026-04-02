import math
import os
import random
import sys
import time
import requests
import zstandard
import pickle as pkl
import numpy as np

import chess
import chess.pgn
import torch
from torch import nn
import tqdm


# ──────────────────────────────────────────────────────────────────────
# Number of input planes:
#   12 piece planes (P N B R Q K × white/black)
#    1 side-to-move plane (all 1s if white to move, all 0s otherwise)
#    4 castling-rights planes (K/Q white, k/q black – full plane 1/0)
#    1 en-passant plane (the target square marked 1)
#   ── total: 18 planes of 8×8
# ──────────────────────────────────────────────────────────────────────
NUM_PLANES = 18


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Encode a FEN string into an 18×8×8 float tensor."""
    board = chess.Board(fen)
    tensor = torch.zeros(NUM_PLANES, 8, 8, dtype=torch.float32)
    piece_to_index = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # --- 12 piece planes ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            idx = piece_to_index[piece.piece_type] + color_offset
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[idx, row, col] = 1.0

    # --- side-to-move plane (plane 12) ---
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # --- castling rights (planes 13-16) ---
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # --- en-passant plane (plane 17) ---
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        tensor[17, row, col] = 1.0

    return tensor


def _mirror_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Colour-flip augmentation: swap white/black pieces, flip ranks,
    swap side-to-move and castling planes."""
    aug = torch.zeros_like(tensor)
    # Swap white pieces (0-5) ↔ black pieces (6-11) AND flip ranks
    aug[0:6] = tensor[6:12].flip(1)
    aug[6:12] = tensor[0:6].flip(1)
    # Invert side-to-move
    aug[12] = 1.0 - tensor[12]
    # Swap castling rights white ↔ black
    aug[13] = tensor[15]
    aug[14] = tensor[16]
    aug[15] = tensor[13]
    aug[16] = tensor[14]
    # Flip en-passant rank
    aug[17] = tensor[17].flip(0)
    return aug


# ──────────────────────────────────────────────────────────────────────
# Residual block used in the convolutional trunk
# ──────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + residual)
        return out


# ──────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation block – cheap global context for each channel
# ──────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))  # global avg pool → (B, C)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.view(b, c, 1, 1)


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = torch.relu(out + residual)
        return out


# ──────────────────────────────────────────────────────────────────────
# Main model – ResNet-style value network
#   input_conv  →  N×SE-ResBlock  →  value head (conv → fc → scalar)
# ──────────────────────────────────────────────────────────────────────
class NNEvaluator(nn.Module):
    def __init__(self, channels: int = 128, num_blocks: int = 8):
        super().__init__()
        # --- trunk ---
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.Sequential(*[SEResBlock(channels) for _ in range(num_blocks)])
        # --- value head ---
        head_channels = 32
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(head_channels * 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # x: (B, 18, 8, 8)
        x = self.input_conv(x)
        x = self.res_tower(x)
        x = self.value_conv(x)
        x = x.view(x.size(0), -1)
        x = self.value_fc(x)
        return x

    def win_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(white wins) ∈ [0, 1]."""
        return torch.sigmoid(self.forward(x))


# ──────────────────────────────────────────────────────────────────────
# Replay buffer for self-play training
# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-capacity FIFO buffer of (position_tensor, game_outcome) pairs."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.data: list[tuple[torch.Tensor, float]] = []

    def add_game(self, positions: list[torch.Tensor], outcome: float, augment: bool = True):
        """Add all positions from one game.

        outcome: 1.0 = white won, 0.0 = black won, 0.5 = draw.
        augment: also store the colour-flipped mirror of each position.
        """
        for t in positions:
            self.data.append((t, outcome))
            if augment:
                self.data.append((_mirror_tensor(t), 1.0 - outcome))
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity :]

    def __len__(self) -> int:
        return len(self.data)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        tensors = torch.stack([t for t, _ in batch])
        targets = torch.tensor([o for _, o in batch], dtype=torch.float32)
        return tensors, targets


# Helper to load Lichess database of real games for initial learning
def load_buffer_from_lichess_db(capacity: int = 200_000) -> ReplayBuffer:
    """Load Lichess game database from data/lichess-games.pkl.

    Returns a list of (FEN, outcome) tuples where outcome ∈ {1.0, 0.5, 0.0}.
    """
    if os.path.exists("data/lichess-games.pkl"):
        print("Loading Lichess database from data/lichess-games.pkl...")
        with open("data/lichess-games.pkl", "rb") as f:
            return pkl.load(f)

    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst"
    print("Downloading Lichess database (111MB compressed)...")
    r = requests.get(url, stream=True)
    dctx = zstandard.ZstdDecompressor()
    with open("data/lichess_db.txt", "wb") as f:
        with dctx.stream_reader(r.raw) as reader:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                f.write(chunk)

    # Parse PGNs lines into (FEN, outcome) tuples
    buffer = ReplayBuffer(capacity=capacity)
    with open("data/lichess_db.txt") as f:
        game_count = 0
        position_count = 0
        skipped_elo = 0

        while position_count < buffer.capacity:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_count += 1

            # Check elo filters
            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")

            # Filter by elo (both players > 2000)
            if not white_elo or not black_elo:
                continue
            try:
                white_elo = int(white_elo)
                black_elo = int(black_elo)
            except ValueError:
                continue

            if white_elo < 2000 or black_elo < 2000:
                skipped_elo += 1
                continue

            # Get game outcome
            result = game.headers.get("Result", "*")
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = 0.0
            elif result == "1/2-1/2":
                outcome = 0.5
            else:
                continue

            # Extract positions from game
            board = chess.Board()
            positions = []
            for move in game.mainline_moves():
                positions.append(fen_to_tensor(board.fen()))
                board.push(move)

            if positions:
                buffer.add_game(positions, outcome)
                position_count += len(positions)

            if game_count % 100 == 0:
                print(f"Loaded {game_count} games ({position_count} positions) from Lichess database...")

        print(
            f"Lichess database loaded: {game_count} games ({position_count} positions, "
            f"{skipped_elo} skipped for elo)"
        )

    if os.path.exists("data/lichess_db.txt"):
        os.remove("data/lichess_db.txt")
    pkl.dump(buffer, open("data/lichess-games.pkl", "wb"))
    return buffer


# ──────────────────────────────────────────────────────────────────────
# Self-play game generation
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def self_play_game(
    evaluator: NNEvaluator,
    initial_fen: str,
    max_moves: int = 150,
    temperature: float = 1.5,
    temp_drop_ply: int = 20,
    resign_threshold: float = 0.03,
    resign_count: int = 10,
    return_draws: bool = True,
) -> tuple[list[torch.Tensor], float]:
    """Play one game of self-play using 1-ply batch evaluation.

    Returns (positions, outcome) where outcome ∈ {1.0, 0.5, 0.0}.
    """
    board = chess.Board(initial_fen)
    positions: list[torch.Tensor] = []
    consec_low = 0

    for ply in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # Record current position
        positions.append(fen_to_tensor(board.fen()))

        # Evaluate every child position in one forward pass
        child_tensors: list[torch.Tensor] = []
        for m in legal_moves:
            board.push(m)
            child_tensors.append(fen_to_tensor(board.fen()))
            board.pop()

        batch = torch.stack(child_tensors)
        logits = evaluator(batch).squeeze(-1)  # white-POV logits

        # Flip so higher = better for side to move
        if board.turn == chess.BLACK:
            logits = -logits

        # Temperature-based move selection
        t = temperature if ply < temp_drop_ply else 0.1
        probs = torch.softmax(logits / max(t, 0.01), dim=0)
        idx = torch.multinomial(probs, 1).item()

        # Early resignation: if best move still looks hopeless
        best_logit = logits.max().item()
        win_prob = 1.0 / (1.0 + math.exp(-max(-20, min(20, best_logit))))
        if win_prob < resign_threshold:
            consec_low += 1
        else:
            consec_low = 0
        if consec_low >= resign_count:
            return positions, (0.0 if board.turn == chess.WHITE else 1.0)

        board.push(legal_moves[idx])

    # Final result
    result_str = board.result()
    if result_str == "1-0":
        return positions, 1.0
    elif result_str == "0-1":
        return positions, 0.0
    return (positions, 0.5) if return_draws else None


# ──────────────────────────────────────────────────────────────────────
# Self-play training loop
# ──────────────────────────────────────────────────────────────────────
def self_play_train(
    num_iterations: int = 30,
    games_per_iter: int = 30,
    epochs_per_iter: int = 5,
    batch_size: int = 256,
    buffer_capacity: int = 200_000,
    lr: float = 1e-4,
    temperature: float = 1.5,
    temp_drop_ply: int = 20,
):
    """Train the evaluator entirely through self-play.

    The model output (raw logit) is trained with BCEWithLogitsLoss
    against game outcomes (1.0 = white win, 0.5 = draw, 0.0 = black win).
    sigmoid(output) gives P(white wins).

    Each iteration:
      1. Play `games_per_iter` games (CPU, 1-ply batch eval).
      2. Store (position, outcome) in a replay buffer.
      3. Train for `epochs_per_iter` on sampled mini-batches.
      4. Save checkpoint.
    """
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Get starting FENs for self-play
    with open("data/puzzles.txt", "r") as f:
        puzzle_fens = [line.split(",")[0].strip() for line in f if line.strip()]
    if not puzzle_fens:
        print("No starting positions found in data/puzzles.txt")
        return
    with open("data/FENs.txt", "r") as f:
        opening_fens = [line.strip() for line in f if line.strip()]
    if not opening_fens:
        print("No starting positions found in data/FENs.txt")
        return

    # Get FENs for accuracy testing during training (comparing eval to stockfish evals)
    with open("data/chessData.csv", "r") as f:
        test_fens = []
        for i, line in enumerate(f):
            if i % 1000 != 0:
                continue
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            fen, stockfish_eval = parts
            try:
                stockfish_eval = float(stockfish_eval) / 100.0  # convert centipawns to pawns
            except ValueError:
                continue
            test_fens.append((fen, stockfish_eval))
            if len(test_fens) >= 1000:
                break
    if not test_fens:
        print("No test positions found in data/chessData.csv")
        return

    evaluator = NNEvaluator(channels=128, num_blocks=8)
    new_evaluator = True
    if os.path.exists("NN_evaluator.pth"):
        try:
            evaluator.load_state_dict(torch.load("NN_evaluator.pth", map_location="cpu", weights_only=True))
            new_evaluator = False
            print("Resumed from existing checkpoint")
        except Exception:
            print("Could not load checkpoint — starting from scratch")

    evaluator = evaluator.to(device)
    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    buffer = ReplayBuffer(buffer_capacity)

    # Learn from actual games to get off the ground, then switch to pure self-play
    if new_evaluator:
        buffer = load_buffer_from_lichess_db(1_000_000)

        print(f"\n{'═' * 60}")
        print(f"  Initial training on Lichess database")
        print(f"{'═' * 60}")
        train_on_replay_buffer(10, 1024, device, evaluator, optimizer, criterion, buffer, 100, test_fens)

        # Save checkpoint before starting self-play
        torch.save(evaluator.state_dict(), "NN_evaluator.pth")
        print(f"  Initial checkpoint saved")

    # Learn via self-play iterations
    for iteration in range(1, num_iterations + 1):
        iter_start = time.perf_counter()
        print(f"\n{'═' * 60}")
        print(f"  Iteration {iteration}/{num_iterations}")
        print(f"{'═' * 60}")

        # ── 1. Self-play on CPU ──────────────────────────────────────
        evaluator.eval()
        cpu_model = NNEvaluator(channels=128, num_blocks=8)
        cpu_model.load_state_dict({k: v.cpu() for k, v in evaluator.state_dict().items()})
        cpu_model.eval()

        results = {"W": 0, "B": 0, "D": 0}
        total_positions = 0

        for i in tqdm.tqdm(range(games_per_iter), desc="Self-play"):
            # 50% of games with puzzle positions, 50% from opening position FENs
            if i % 2 == 0:
                # Play until we get a non-draw result (engine is too weak to learn much from draws)
                self_play_result = None
                while self_play_result is None:
                    self_play_result = self_play_game(
                        cpu_model,
                        random.choice(puzzle_fens),
                        max_moves=150,
                        temperature=temperature,
                        temp_drop_ply=temp_drop_ply,
                        return_draws=False,
                    )
            else:
                # We allow draws from the opening positions
                self_play_result = self_play_game(
                    cpu_model,
                    random.choice(opening_fens),
                    max_moves=150,
                    temperature=temperature,
                    temp_drop_ply=temp_drop_ply,
                )

            positions, outcome = self_play_result
            buffer.add_game(positions, outcome)
            total_positions += len(positions)
            if outcome == 1.0:
                results["W"] += 1
            elif outcome == 0.0:
                results["B"] += 1
            else:
                results["D"] += 1

        print(
            f"  Results  W {results['W']}  B {results['B']}  "
            f"D {results['D']}  |  +{total_positions} positions  "
            f"|  buffer {len(buffer)}"
        )

        # ── 2. Train on replay buffer ────────────────────────────────
        if len(buffer) < batch_size:
            print("  Buffer too small, skipping training")
            continue

        evaluator.train()
        steps_per_epoch = max(1, len(buffer) // batch_size)

        train_on_replay_buffer(
            epochs_per_iter, batch_size, device, evaluator, optimizer, criterion, buffer, steps_per_epoch, test_fens
        )

        # ── 3. Save checkpoint ───────────────────────────────────────
        torch.save(evaluator.state_dict(), "NN_evaluator.pth")
        elapsed = time.perf_counter() - iter_start
        print(f"  Checkpoint saved  ({elapsed:.1f}s)")


def train_on_replay_buffer(
    epochs_per_iter: int,
    batch_size: int,
    device: torch.device,
    evaluator: NNEvaluator,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    buffer: ReplayBuffer,
    steps_per_epoch: int,
    test_fens: list[tuple[str, float]],
):
    for epoch in range(1, epochs_per_iter + 1):
        evaluator.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            tensors, targets = buffer.sample_batch(batch_size)
            tensors = tensors.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = evaluator(tensors).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate on test positions
        # Since we compare against stockfish (centipawn evals):
        # eval > 2.0 means white is better, so P(white wins) should be > 0.5
        # eval < -2.0 means black is better, so P(white wins) should be < 0.5
        # -2.0 <= eval <= 2.0 means roughly equal, so P(white wins) should be 0.5±0.1
        evaluator.eval()
        successful = 0
        predictions = []
        targets = []
        for fen, stockfish_eval in test_fens:
            t = fen_to_tensor(fen).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = evaluator(t).item()
                wp = 1.0 / (1.0 + math.exp(-max(-20, min(20, logit))))

            # Accuracy metric
            if (
                (stockfish_eval > 2.0 and wp > 0.5)
                or (stockfish_eval < -2.0 and wp < 0.5)
                or (-2.0 <= stockfish_eval <= 2.0 and 0.4 <= wp <= 0.6)
            ):
                successful += 1

            # Correlation metric
            predictions.append(wp)
            # Normalize stockfish eval to [0, 1] with sigmoid
            targets.append(1 / (1 + math.exp(-stockfish_eval / 2)))

        avg = epoch_loss / steps_per_epoch
        acc = successful / len(test_fens)
        correlation = np.corrcoef(predictions, targets)[0, 1]
        print(f"    Epoch {epoch}/{epochs_per_iter}  loss {avg:.4f}  acc {acc:.1%}  corr {correlation:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> NNEvaluator:
    evaluator = NNEvaluator()
    evaluator.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    evaluator.eval()
    return evaluator


def play(board: chess.Board, millis: int, evaluator: NNEvaluator) -> dict:
    """Run a simple minimax search with alpha-beta pruning."""
    engine = SearchEngine(board, evaluator, millis)
    return engine.search()


# ──────────────────────────────────────────────────────────────────────
# Simple alpha-beta configuration
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_SEARCH_DEPTH = 60
_DEFAULT_QUIESCENCE_DEPTH = 6


class SearchEngine:
    """Minimal search context for one `play()` call.

    Uses alpha-beta with scores as P(white wins) in [0, 1].
    White maximizes score, Black minimizes score.
    """

    def __init__(
        self,
        board: chess.Board,
        evaluator: torch.Module,
        millis: int,
        neg_inf: float = 0.0,
        pos_inf: float = 1.0,
        draw_score: float = 0.5,
    ):
        self.root_board = board.copy()
        self.evaluator = evaluator
        self.millis = millis
        self.use_time_limit = millis > 0
        self.deadline: float = 0.0
        self.nodes: int = 0
        self.qnodes: int = 0
        self.neg_inf = neg_inf
        self.pos_inf = pos_inf
        self.draw_score = draw_score

    # ── fast NN eval ───────────────────────────────────────────────
    def evaluate(self, board: chess.Board) -> float:
        """Return P(white wins) ∈ [0, 1]."""
        t = fen_to_tensor(board.fen()).unsqueeze(0)
        with torch.no_grad():
            return self.evaluator.win_probability(t).item()

    def is_time_up(self) -> bool:
        if not self.use_time_limit:
            return False
        return time.perf_counter() >= self.deadline

    def terminal_score(self, board: chess.Board) -> float | None:
        """Return terminal score in white-win probability, or None."""
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return self.draw_score
        return self.pos_inf if outcome.winner == chess.WHITE else self.neg_inf

    def noisy_moves(self, board: chess.Board) -> list[chess.Move]:
        """Generate tactical moves for quiescence search."""
        moves: list[chess.Move] = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion is not None or board.gives_check(move):
                moves.append(move)
        return moves

    def quiescence(self, board: chess.Board, alpha: float, beta: float, qdepth: int) -> float:
        """Search only tactical continuations to stabilize leaf evaluations."""
        self.qnodes += 1

        terminal = self.terminal_score(board)
        if terminal is not None:
            return terminal
        if qdepth <= 0 or self.is_time_up():
            return self.evaluate(board)

        stand_pat = self.evaluate(board)
        moves = self.noisy_moves(board)
        if not moves:
            return stand_pat

        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

            for move in moves:
                board.push(move)
                score = self.quiescence(board, alpha, beta, qdepth - 1)
                board.pop()

                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            return alpha

        if stand_pat <= alpha:
            return stand_pat
        if stand_pat < beta:
            beta = stand_pat

        for move in moves:
            board.push(move)
            score = self.quiescence(board, alpha, beta, qdepth - 1)
            board.pop()

            if score < beta:
                beta = score
            if alpha >= beta:
                break
        return beta

    # ── plain alpha-beta minimax ─────────────────────────────────────
    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Alpha-beta with white maximizing and black minimizing."""
        self.nodes += 1
        terminal = self.terminal_score(board)
        if terminal is not None:
            return terminal
        if self.is_time_up():
            return self.evaluate(board)
        if depth <= 0:
            return self.quiescence(board, alpha, beta, _DEFAULT_QUIESCENCE_DEPTH)

        if board.turn == chess.WHITE:
            best_score = self.neg_inf
            for move in board.legal_moves:
                board.push(move)
                score = self.alpha_beta(board, depth - 1, alpha, beta)
                board.pop()

                if score > best_score:
                    best_score = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            return best_score

        best_score = self.pos_inf
        for move in board.legal_moves:
            board.push(move)
            score = self.alpha_beta(board, depth - 1, alpha, beta)
            board.pop()

            if score < best_score:
                best_score = score
            if score < beta:
                beta = score
            if alpha >= beta:
                break
        return best_score

    # ── iterative deepening root search ──────────────────────────────
    def search(self) -> dict:
        if self.evaluator is not None:
            prev_training_mode = self.evaluator.training
            self.evaluator.eval()

        start = time.perf_counter()
        self.deadline = start + self.millis / 1000.0 if self.use_time_limit else float("inf")
        # For timed search, rely on deadline checks instead of a tiny fixed depth cap.
        # This keeps iterative deepening truly iterative under common controls like 1s.
        max_depth = _DEFAULT_SEARCH_DEPTH

        legal = list(self.root_board.legal_moves)
        if len(legal) == 0:
            return {"move": None, "eval": 0, "time": 0, "depth": 0, "nodes": 0}
        best_move = legal[0]
        best_score = self.evaluate(self.root_board)
        completed_depth = 0

        try:
            for depth in range(1, max_depth + 1):
                if self.is_time_up():
                    break

                alpha = self.neg_inf
                beta = self.pos_inf
                iter_moves = legal.copy()
                # Principal-variation style ordering: try last best root move first.
                if best_move in iter_moves:
                    iter_moves.remove(best_move)
                    iter_moves.insert(0, best_move)

                iter_best_move = iter_moves[0]
                iter_best_score = self.neg_inf if self.root_board.turn == chess.WHITE else self.pos_inf
                searched_any_root_move = False
                fully_completed_iteration = True

                for move in iter_moves:
                    if self.is_time_up():
                        fully_completed_iteration = False
                        break
                    self.root_board.push(move)
                    score = self.alpha_beta(self.root_board, depth - 1, alpha, beta)
                    self.root_board.pop()
                    searched_any_root_move = True

                    if self.root_board.turn == chess.WHITE:
                        if score > iter_best_score:
                            iter_best_score = score
                            iter_best_move = move
                        if score > alpha:
                            alpha = score
                    else:
                        if score < iter_best_score:
                            iter_best_score = score
                            iter_best_move = move
                        if score < beta:
                            beta = score

                    if alpha >= beta:
                        break

                # Only trust a deeper iteration if it fully completed.
                # A partial iteration can search just a subset of root moves,
                # which may overwrite a stable shallower best move with noise.
                if searched_any_root_move:
                    if fully_completed_iteration:
                        best_move = iter_best_move
                        best_score = iter_best_score
                        completed_depth = depth
                    elif completed_depth == 0:
                        # No completed iteration yet: keep best seen so far
                        # from this partial pass to avoid defaulting blindly.
                        best_move = iter_best_move
                        best_score = iter_best_score

            elapsed = time.perf_counter() - start
            print(
                f"Time elapsed: {elapsed:.3f}s  Depth reached: {completed_depth}  "
                f"Nodes searched: {self.nodes:,} (+q {self.qnodes:,})  Eval: {best_score:.4f}"
            )
            return {
                "move": best_move.uci(),
                "eval": round(best_score, 4),
                "time": round(elapsed, 3),
                "depth": completed_depth,
                "nodes": self.nodes,
            }
        finally:
            if self.evaluator is not None:
                self.evaluator.train(prev_training_mode)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        n_iter = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
        self_play_train(num_iterations=n_iter)
    elif os.path.exists("NN_evaluator.pth") and len(sys.argv) == 2:
        model = load_model("NN_evaluator.pth")
        t = fen_to_tensor(sys.argv[1]).unsqueeze(0)
        with torch.no_grad():
            logit = model(t).item()
            wp = 1.0 / (1.0 + math.exp(-max(-20, min(20, logit))))
        print(f"NN logit: {logit:.4f}  P(white wins): {wp:.1%}")

        # Evaluate every legal moves resulting position and print the 3 best ones
        board = chess.Board(sys.argv[1])
        move_scores = {}
        for move in board.legal_moves:
            board.push(move)
            t = fen_to_tensor(board.fen()).unsqueeze(0)
            with torch.no_grad():
                logit = model(t).item()
                wp = 1.0 / (1.0 + math.exp(-max(-20, min(20, logit))))
            move_scores[move.uci()] = wp
            board.pop()

        best_moves = sorted(
            move_scores.items(), key=lambda x: x[1] if board.turn == chess.WHITE else 1 - x[1], reverse=True
        )[:3]
        print("\nTop moves:")
        for move, score in best_moves:
            print(f"  {move}: P(white wins) = {score:.1%}")

    else:
        self_play_train()
