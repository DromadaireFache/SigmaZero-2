import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import chess
import chess.polyglot
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
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
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
        w = x.mean(dim=(2, 3))             # global avg pool → (B, C)
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
class NNUEEvaluator(nn.Module):
    def __init__(self, channels: int = 128, num_blocks: int = 8):
        super().__init__()
        # --- trunk ---
        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.Sequential(
            *[SEResBlock(channels) for _ in range(num_blocks)]
        )
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

    def add_game(self, positions: list[torch.Tensor], outcome: float,
                 augment: bool = True):
        """Add all positions from one game.

        outcome: 1.0 = white won, 0.0 = black won, 0.5 = draw.
        augment: also store the colour-flipped mirror of each position.
        """
        for t in positions:
            self.data.append((t, outcome))
            if augment:
                self.data.append((_mirror_tensor(t), 1.0 - outcome))
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]

    def __len__(self) -> int:
        return len(self.data)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        tensors = torch.stack([t for t, _ in batch])
        targets = torch.tensor([o for _, o in batch], dtype=torch.float32)
        return tensors, targets


# ──────────────────────────────────────────────────────────────────────
# Self-play game generation
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def self_play_game(
    evaluator: NNUEEvaluator,
    max_moves: int = 400,
    temperature: float = 1.5,
    temp_drop_ply: int = 20,
    resign_threshold: float = 0.03,
    resign_count: int = 10,
) -> tuple[list[torch.Tensor], float]:
    """Play one game of self-play using 1-ply batch evaluation.

    Returns (positions, outcome) where outcome ∈ {1.0, 0.5, 0.0}.
    """
    board = chess.Board()
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
        logits = evaluator(batch).squeeze(-1)      # white-POV logits

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
    result_str = board.result(claim_draw=True)
    if result_str == "1-0":
        return positions, 1.0
    elif result_str == "0-1":
        return positions, 0.0
    return positions, 0.5


# ──────────────────────────────────────────────────────────────────────
# Self-play training loop
# ──────────────────────────────────────────────────────────────────────
def self_play_train(
    num_iterations: int = 100,
    games_per_iter: int = 100,
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
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    evaluator = NNUEEvaluator(channels=128, num_blocks=8)
    if os.path.exists("nnue_evaluator.pth"):
        try:
            evaluator.load_state_dict(
                torch.load("nnue_evaluator.pth", map_location="cpu",
                           weights_only=True))
            print("Resumed from existing checkpoint")
        except Exception:
            print("Could not load checkpoint — starting from scratch")

    evaluator = evaluator.to(device)
    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=lr,
                                  weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    buffer = ReplayBuffer(buffer_capacity)

    for iteration in range(1, num_iterations + 1):
        iter_start = time.perf_counter()
        print(f"\n{'═' * 60}")
        print(f"  Iteration {iteration}/{num_iterations}")
        print(f"{'═' * 60}")

        # ── 1. Self-play on CPU ──────────────────────────────────────
        evaluator.eval()
        cpu_model = NNUEEvaluator(channels=128, num_blocks=8)
        cpu_model.load_state_dict(
            {k: v.cpu() for k, v in evaluator.state_dict().items()})
        cpu_model.eval()

        results = {"W": 0, "B": 0, "D": 0}
        total_positions = 0

        for _ in tqdm.tqdm(range(games_per_iter), desc="Self-play"):
            positions, outcome = self_play_game(
                cpu_model, max_moves=400,
                temperature=temperature, temp_drop_ply=temp_drop_ply)
            buffer.add_game(positions, outcome, augment=True)
            total_positions += len(positions)
            if outcome == 1.0:
                results["W"] += 1
            elif outcome == 0.0:
                results["B"] += 1
            else:
                results["D"] += 1

        print(f"  Results  W {results['W']}  B {results['B']}  "
              f"D {results['D']}  |  +{total_positions} positions  "
              f"|  buffer {len(buffer)}")

        # ── 2. Train on replay buffer ────────────────────────────────
        if len(buffer) < batch_size:
            print("  Buffer too small, skipping training")
            continue

        evaluator.train()
        steps_per_epoch = max(1, len(buffer) // batch_size)

        for epoch in range(1, epochs_per_iter + 1):
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

            avg = epoch_loss / steps_per_epoch
            print(f"    Epoch {epoch}/{epochs_per_iter}  loss {avg:.4f}")

        # ── 3. Save checkpoint ───────────────────────────────────────
        torch.save(evaluator.state_dict(), "nnue_evaluator.pth")
        elapsed = time.perf_counter() - iter_start
        print(f"  Checkpoint saved  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> NNUEEvaluator:
    evaluator = NNUEEvaluator()
    evaluator.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    evaluator.eval()
    return evaluator


def play(board: chess.Board, millis: int, evaluator: NNUEEvaluator) -> dict:
    """Iterative-deepening alpha-beta search with the NNUE evaluator.

    Features:
      - Iterative deepening with time control (`millis` budget).
      - Alpha-beta pruning with principal-variation move ordering.
      - Transposition table (hash-map on Zobrist hash from python-chess).
      - MVV-LVA move ordering for captures, killer/history heuristic.
      - Quiescence search for captures to avoid horizon effects.
      - Multithreaded root-move evaluation (Lazy SMP style).
    """
    engine = _SearchEngine(board, evaluator, millis)
    return engine.search()


# ──────────────────────────────────────────────────────────────────────
# Transposition-table entry
# ──────────────────────────────────────────────────────────────────────
FLAG_EXACT = 0
FLAG_LOWER = 1   # beta cut-off  (score >= beta)
FLAG_UPPER = 2   # failed low    (score <= alpha)


@dataclass(slots=True)
class TTEntry:
    key: int
    depth: int
    score: float
    flag: int
    best_move: chess.Move | None


# ──────────────────────────────────────────────────────────────────────
# Piece values for MVV-LVA ordering
# ──────────────────────────────────────────────────────────────────────
_PIECE_VAL = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

_MATE_SCORE = 100_000.0
_MAX_DEPTH  = 64


class _SearchEngine:
    """Self-contained search context for one `play()` call.

    All scores are P(side-to-move wins) ∈ [0, 1].
    Negamax negation uses  1 - score  instead of  -score.
    Mate: 1.0 (side-to-move wins); 0.0 minus a small ply bonus (loss).
    Draw: 0.5.
    """

    def __init__(self, board: chess.Board, evaluator: NNUEEvaluator,
                 millis: int):
        self.root_board = board.copy()
        self.evaluator = evaluator
        self.millis = millis
        self.deadline: float = 0.0
        self.nodes: int = 0
        # Transposition table  (shared across threads via GIL)
        self.tt: dict[int, TTEntry] = {}
        # Killer moves  [depth] → list of up to 2 moves
        self.killers: list[list[chess.Move]] = [[] for _ in range(_MAX_DEPTH)]
        # History heuristic  (from_sq, to_sq) → score
        self.history: dict[tuple[int, int], int] = {}

    # ── fast NNUE eval ───────────────────────────────────────────────
    def _evaluate(self, board: chess.Board) -> float:
        """Return P(side-to-move wins) ∈ [0, 1].

        sigmoid(raw_logit) gives P(white wins); we flip for black.
        """
        t = fen_to_tensor(board.fen()).unsqueeze(0)
        with torch.no_grad():
            wp = self.evaluator.win_probability(t).item()
        return wp if board.turn == chess.WHITE else 1.0 - wp

    def _is_time_up(self) -> bool:
        return time.perf_counter() >= self.deadline

    # ── move ordering ────────────────────────────────────────────────
    def _order_moves(self, board: chess.Board, tt_move: chess.Move | None,
                     depth: int) -> list[chess.Move]:
        """Sort moves: TT move → captures (MVV-LVA) → killers → history."""
        moves = list(board.legal_moves)
        scores: list[tuple[int, chess.Move]] = []

        for m in moves:
            s = 0
            if m == tt_move:
                s = 1_000_000
            elif board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or chess.PAWN
                attacker = board.piece_type_at(m.from_square) or chess.PAWN
                s = 100_000 + _PIECE_VAL[victim] * 10 - _PIECE_VAL[attacker]
            elif depth < _MAX_DEPTH and m in self.killers[depth]:
                s = 50_000
            else:
                s = self.history.get((m.from_square, m.to_square), 0)
            scores.append((s, m))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scores]

    # ── quiescence search ────────────────────────────────────────────
    def _quiesce(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Quiescence: extend captures at leaf nodes. Scores in [0, 1]."""
        self.nodes += 1

        stand_pat = self._evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Only search captures
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        # MVV-LVA sort
        captures.sort(
            key=lambda m: _PIECE_VAL.get(
                board.piece_type_at(m.to_square) or chess.PAWN, 0
            ) * 10 - _PIECE_VAL.get(
                board.piece_type_at(m.from_square) or chess.PAWN, 0
            ),
            reverse=True,
        )

        for m in captures:
            if self._is_time_up():
                return alpha
            board.push(m)
            score = 1.0 - self._quiesce(board, 1.0 - beta, 1.0 - alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ── alpha-beta with TT ───────────────────────────────────────────
    def _alpha_beta(self, board: chess.Board, depth: int,
                    alpha: float, beta: float, ply: int) -> float:
        """Negamax alpha-beta. All scores are P(side-to-move wins) ∈ [0,1]."""
        self.nodes += 1
        alpha_orig = alpha

        # ── terminal checks ──────────────────────────────────────────
        if board.is_checkmate():
            # Side to move is checkmated → they lose.
            # Small ply bonus so shorter mates are preferred.
            return max(0.0, 0.0 + ply * 1e-6)
        if board.is_stalemate() or board.is_insufficient_material() \
                or board.is_fifty_moves() or board.is_repetition(2):
            return 0.5

        # ── transposition-table probe ────────────────────────────────
        key = chess.polyglot.zobrist_hash(board)
        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None and entry.key == key and entry.depth >= depth:
            tt_move = entry.best_move
            if entry.flag == FLAG_EXACT:
                return entry.score
            elif entry.flag == FLAG_LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == FLAG_UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score
        elif entry is not None:
            tt_move = entry.best_move

        # ── leaf / quiescence ────────────────────────────────────────
        if depth <= 0:
            return self._quiesce(board, alpha, beta)

        # ── null-move pruning (reduction = 3) ────────────────────────
        if depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            null_score = 1.0 - self._alpha_beta(
                board, depth - 3, 1.0 - beta, 1.0 - beta + 0.01, ply + 1)
            board.pop()
            if null_score >= beta:
                return beta

        # ── recurse over children ────────────────────────────────────
        best_score = 0.0     # worst possible for side to move
        best_move: chess.Move | None = None
        moves = self._order_moves(board, tt_move, ply)

        for i, m in enumerate(moves):
            if self._is_time_up():
                break

            is_capture = board.is_capture(m)
            board.push(m)
            gives_check = board.is_check()

            # ── late-move reduction ──────────────────────────────────
            if i >= 4 and depth >= 3 and not is_capture \
                    and not gives_check:
                score = 1.0 - self._alpha_beta(
                    board, depth - 2,
                    1.0 - alpha - 0.01, 1.0 - alpha, ply + 1)
                if score <= alpha:
                    board.pop()
                    continue
                # Re-search at full depth if it looked promising
            score = 1.0 - self._alpha_beta(
                board, depth - 1, 1.0 - beta, 1.0 - alpha, ply + 1)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = m
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Update killer moves
                if not is_capture and ply < _MAX_DEPTH:
                    killers = self.killers[ply]
                    if m not in killers:
                        if len(killers) >= 2:
                            killers.pop(0)
                        killers.append(m)
                # Update history heuristic
                self.history[(m.from_square, m.to_square)] = \
                    self.history.get((m.from_square, m.to_square), 0) + depth * depth
                break

        # ── store in TT ──────────────────────────────────────────────
        if best_move is not None:
            if best_score <= alpha_orig:
                flag = FLAG_UPPER
            elif best_score >= beta:
                flag = FLAG_LOWER
            else:
                flag = FLAG_EXACT
            self.tt[key] = TTEntry(key, depth, best_score, flag, best_move)

        return best_score

    # ── root search for a single move (used by threads) ──────────────
    def _search_root_move(self, move: chess.Move, depth: int,
                          alpha: float, beta: float) -> tuple[chess.Move, float]:
        """Search a single root move on a private board copy."""
        board = self.root_board.copy()
        board.push(move)
        score = 1.0 - self._alpha_beta(
            board, depth - 1, 1.0 - beta, 1.0 - alpha, 1)
        board.pop()
        return move, score

    # ── iterative deepening with multithreaded root ──────────────────
    def search(self) -> dict:
        start = time.perf_counter()
        self.deadline = start + self.millis / 1000.0

        best_move: chess.Move | None = None
        best_score = 0.5
        completed_depth = 0

        # Quick check: if only one legal move, return immediately
        legal = list(self.root_board.legal_moves)
        if len(legal) == 0:
            return {"move": None, "eval": 0, "time": 0, "depth": 0, "nodes": 0}
        if len(legal) == 1:
            with torch.no_grad():
                self.root_board.push(legal[0])
                stm_wp = self._evaluate(self.root_board)
                self.root_board.pop()
            # Convert to P(white wins)
            wp = stm_wp if self.root_board.turn == chess.WHITE else 1.0 - stm_wp
            return {
                "move": legal[0].uci(), "eval": round(wp, 4),
                "time": time.perf_counter() - start, "depth": 1, "nodes": 1,
            }

        # PV move from previous iteration for ordering
        pv_move: chess.Move | None = None
        n_threads = min(len(legal), max(1, os.cpu_count() or 1))

        for depth in range(1, _MAX_DEPTH + 1):
            if self._is_time_up():
                break

            alpha = 0.0
            beta  = 1.0
            iter_best_move: chess.Move | None = None
            iter_best_score = 0.0

            # Order root moves: PV first, then captures, then rest
            root_moves = self._order_moves(self.root_board, pv_move, 0)

            # ── multithreaded root search ────────────────────────────
            if depth >= 3 and n_threads > 1:
                # Search PV move first with full window (sequential)
                if root_moves:
                    pv_candidate = root_moves[0]
                    _, pv_score = self._search_root_move(
                        pv_candidate, depth, alpha, beta)
                    if pv_score > iter_best_score:
                        iter_best_score = pv_score
                        iter_best_move = pv_candidate
                    alpha = max(alpha, pv_score)

                # Remaining moves in parallel with null-window + re-search
                remaining = root_moves[1:]
                with ThreadPoolExecutor(max_workers=n_threads) as pool:
                    futures = {
                        pool.submit(
                            self._search_root_move, m, depth,
                            alpha, alpha + 0.01,
                        ): m for m in remaining
                    }
                    for fut in as_completed(futures):
                        if self._is_time_up():
                            break
                        m, score = fut.result()
                        if score > alpha:
                            # Re-search with full window
                            _, score = self._search_root_move(
                                m, depth, alpha, beta)
                        if score > iter_best_score:
                            iter_best_score = score
                            iter_best_move = m
                            alpha = max(alpha, score)
            else:
                # Sequential search at low depths
                for m in root_moves:
                    if self._is_time_up():
                        break
                    _, score = self._search_root_move(m, depth, alpha, beta)
                    if score > iter_best_score:
                        iter_best_score = score
                        iter_best_move = m
                    alpha = max(alpha, score)

            # If we completed this depth, record the result
            if not self._is_time_up() and iter_best_move is not None:
                best_move = iter_best_move
                best_score = iter_best_score
                completed_depth = depth
                pv_move = best_move

        # Convert from side-to-move win prob → P(white wins) for display
        if self.root_board.turn == chess.WHITE:
            display_eval = best_score
        else:
            display_eval = 1.0 - best_score

        elapsed = time.perf_counter() - start
        return {
            "move": best_move.uci() if best_move else None,
            "eval": round(display_eval, 4),
            "time": round(elapsed, 3),
            "depth": completed_depth,
            "nodes": self.nodes,
        }


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        n_iter = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
        self_play_train(num_iterations=n_iter)
    elif os.path.exists("nnue_evaluator.pth") and len(sys.argv) == 2:
        model = load_model("nnue_evaluator.pth")
        t = fen_to_tensor(sys.argv[1]).unsqueeze(0)
        with torch.no_grad():
            logit = model(t).item()
            wp = 1.0 / (1.0 + math.exp(-max(-20, min(20, logit))))
        print(f"NNUE logit: {logit:.4f}  P(white wins): {wp:.1%}")
    else:
        self_play_train()