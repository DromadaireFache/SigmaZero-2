from pprint import pprint
import sys
import subprocess
import re
import chess
import pytest

try:
    import sigma_zero
except ModuleNotFoundError:
    from . import sigma_zero

DEPTH = 4
STOCKFISH_EXE = "stockfish"


def stockfish_perft(fen: str, depth: int, stockfish_path: str = STOCKFISH_EXE) -> int:
    """Use Stockfish's perft command to count nodes at a given depth."""
    process = subprocess.Popen(
        stockfish_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    commands = [
        "uci",
        "isready",
        f"position fen {fen}",
        f"go perft {depth}",
    ]

    for cmd in commands:
        process.stdin.write(cmd + "\n")
        process.stdin.flush()

    # Collect output until we see the final node count
    output_lines: list[str] = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output_lines.append(line.strip())
        # Perft output ends with "Nodes searched: <number>"
        if line.startswith("Nodes searched:"):
            break

    # Terminate the process
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.terminate()

    # Parse the node count from the last line
    for line in reversed(output_lines):
        if line.startswith("Nodes searched:"):
            match = re.search(r"Nodes searched:\s*(\d+)", line)
            if match:
                return int(match.group(1))

    return -1


def count_moves(board: chess.Board, depth: int):
    """Count moves using Stockfish perft."""
    return stockfish_perft(board.fen(), depth)


def get_puzzle_fens():
    with open("data/puzzles.txt", "r") as f:
        return [line.strip().split(",", 1)[0] for line in f if line.strip()]


@pytest.mark.parametrize("fen", get_puzzle_fens())
def test_move_generation(fen: str):
    mybot = sigma_zero.moves(fen, DEPTH).get("nodes", -1)
    chesslib = count_moves(chess.Board(fen), DEPTH)
    assert mybot == chesslib


def find_wrong_move_generation(board: chess.Board, depth: int, search_depth: int = 1):
    sf_moves = count_moves(board, search_depth)
    n_moves = sigma_zero.moves(board.fen(), search_depth).get("nodes", -1)
    if sf_moves != n_moves:
        print(f"Mismatch found at depth {depth} for FEN:\n{board.fen()}")
        print(f"Stockfish moves: {sf_moves}, SigmaZero moves: {n_moves}")
        sys.exit(1)

    if depth == 0:
        return

    for move in board.legal_moves:
        board.push(move)
        find_wrong_move_generation(board, depth - 1, search_depth)
        board.pop()


if __name__ == "__main__":
    #     pprint(sigma_zero.old_play("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 10000))
    #     pprint(sigma_zero.play("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2000))

    FEN = "rnbqk2r/ppppnpp1/4P2p/8/1P6/b7/P1P1PPPP/RNBQKBNR w KQkq - 1 2"
    DEPTH = 6

    print("----- Stockfish -----")
    print("total =", count_moves(chess.Board(FEN), DEPTH))
    print("\n----- SigmaZero 2 -----")
    print("mybot = ", sigma_zero.moves(FEN, DEPTH).get("nodes", -1))

    find_wrong_move_generation(chess.Board(FEN), DEPTH)
    print("No mismatches found")

    # board = chess.Board(FEN)
    # for move in board.legal_moves:
    #     board.push(move)
    #     fen = board.fen()
    #     mybot = sigma_zero.moves(fen, DEPTH - 1)
    #     if not mybot:
    #         print(fen)
    #         print(move)
    #     board.pop()
