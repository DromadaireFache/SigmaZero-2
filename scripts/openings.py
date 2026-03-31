from pprint import pprint
import subprocess
import re
from dataclasses import dataclass
import sigma_zero

import chess


STOCKFISH_EXE = (
    r"C:\Users\charl\Downloads\stockfish-windows-x86-64-avx2 (1)\stockfish\stockfish-windows-x86-64-avx2.exe"
)


@dataclass
class MoveAnalysis:
    rank: int
    move: str
    score: float | str


def stockfish_best_moves(fen, stockfish_path, depth=20, multipv=3) -> list[MoveAnalysis]:
    # Start Stockfish as a process we can communicate with
    process = subprocess.Popen(
        stockfish_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1  # Line buffered
    )

    # Send setup commands
    commands = [
        "uci",
        "isready",
        f"setoption name MultiPV value {multipv}",
        f"position fen {fen}",
        f"go depth {depth}",
    ]

    for cmd in commands:
        process.stdin.write(cmd + "\n")
        process.stdin.flush()

    # Collect output until we see "bestmove" which signals analysis completion
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        output_lines.append(line)
        if line.startswith("bestmove"):
            break

    # Now terminate the process
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.terminate()

    # Join all lines for regex processing
    output = "".join(output_lines)

    # Capture the most recent multipv analysis for each rank
    latest_analyses = {}

    # Look for info lines with multipv, score, and pv
    info_regex = re.compile(r"info.*?multipv (\d+).*?score (cp|mate) (-?\d+).*?pv (.+?)(?=\n|$)")

    for match in info_regex.finditer(output):
        rank = int(match.group(1))
        eval_type = match.group(2)
        eval_value = int(match.group(3))
        moves = match.group(4).split()[0]  # Just take the first move

        if eval_type == "cp":
            score = eval_value / 100.0
        else:
            score = f"mate {eval_value}"

        latest_analyses[rank] = MoveAnalysis(rank, moves, score)

    # Convert to list and sort by rank
    moves = list(latest_analyses.values())
    moves.sort(key=lambda x: x.rank)

    return moves


def openings_database(depth: int, board=chess.Board(), db={}) -> dict[str, list[str]]:
    best_moves = stockfish_best_moves(board.fen(), STOCKFISH_EXE, multipv=min(depth, 3))
    zhash = sigma_zero.zhash(board.fen())
    db[zhash] = [move.move for move in best_moves]

    if depth == 1:
        return db

    for move_analysis in best_moves:
        move = chess.Move.from_uci(move_analysis.move)
        board.push(move)
        openings_database(depth - 1, board, db)
        board.pop()

    return db


# Example usage:
if __name__ == "__main__":
    db = openings_database(4)

    with open("openings.db", "w") as f:
        for fen, moves in db.items():
            f.write(f"{fen},{len(moves)},{','.join(moves)}\n")
