import os
import subprocess
import json
import sys
from typing import Any
from itertools import permutations


DEPTH = 5
_COMPILED = set()
if sys.platform == "linux":
    EXE_FILE = "./sigma-zero"
    OLD_EXE_FILE = "./old"
else:
    EXE_FILE = "sigma-zero.exe"
    OLD_EXE_FILE = "old.exe"
FENS = [
    "r1bq1rk1/p5pp/1p3b2/2pp4/1n1Pp3/1P2P1P1/1B1Q1PBP/2R1NRK1 b - - 1 17",
    "8/6k1/5q1p/1p6/8/7P/6P1/Q6K w - - 11 62",
    "rnbqk2r/ppppnpp1/4P2p/8/1P6/b7/P1P1PPPP/RNBQKBNR w KQkq - 1 2",
    "6k1/5pp1/PbN1p2p/2rn1b2/Q7/6P1/5PKP/5B2 b - - 10 39",
    "2kr1n2/5p2/pq4p1/1p2Pp2/8/5NR1/PPP1QPP1/1K6 w - - 0 23",
    "8/R7/p1k1p3/1p3p2/8/P7/KPP2PP1/3r4 b - - 4 31",
    "rnbq1rk1/ppp2pp1/3b1n1p/3p4/3P4/2NB4/PPP1NPPP/R1BQ1RK1 w - - 0 8",
    "4r1k1/p1qb1pp1/1p3n1p/2pp4/3P1N2/2PQ1P2/PP2R1PP/1B4K1 b - - 2 21",
]


def make(version: str | None = None):
    global _COMPILED
    cmd = f"make {version}" if version else "make"
    if cmd in _COMPILED:
        return
    if os.system(cmd) != 0:
        sys.exit(1)
    _COMPILED.add(cmd)


def command(cmd: str, JSON: bool = False, exe: str = EXE_FILE) -> str | Any:
    make()
    result = subprocess.run(rf"{exe} {cmd}", shell=True, capture_output=True, text=True)
    if JSON:
        try:
            return json.loads(result.stdout)
        except:
            print(result.stdout)
            return {"error": "Failed to parse JSON", "raw": result.stdout}
    else:
        return result


def help():
    command("help")


def version():
    command("version")


def moves(fen: str, depth: int) -> dict:
    return dict(command(f'moves "{fen}" "{depth}"', JSON=True))


def play(fen: str, millis: int) -> dict:
    return dict(command(f'play "{fen}" "{millis}"', JSON=True))


def old_play(fen: str, millis: int) -> dict:
    result = dict(command(f'play "{fen}" "{millis}"', JSON=True, exe=OLD_EXE_FILE))
    return result


def time_move_gen() -> tuple[float, float]:
    total = 0
    nps = 0
    for i, fen in enumerate(FENS):
        result = moves(fen, DEPTH)
        if result.get("nodes", -1) != -1:
            time = result.get("time", 0)
            print(f"{i+1} {time:.2f}")
            total += time
            nps += result.get("nps", 0) / len(FENS)
    print(f"total {total:.2f}")
    print(f"nps   {nps:.0f}")
    return total, nps


if __name__ == "__main__":
    time_move_gen()

    # LIST = ["PIECE_PAWN", "PIECE_KNIGHT", "PIECE_KING", "PIECE_BISHOP", "PIECE_ROOK", "PIECE_QUEEN"]
    # LIST = ["PAWN_ATTACKS", "KNIGHT_ATTACKS", "KING_ATTACKS", "BISHOP_ATTACKS", "ROOK_ATTACKS"]
    # best_nps = float("-inf")
    # best_order = None
    # for perm in permutations(LIST):
    #     with open("src/main.c", "r") as f:
    #         content = f.read()
    #     with open("src/main.c", "w") as f:
    #         # content_new = content.replace("// TEST PIECES HERE", "\n    else ".join(perm))
    #         content_new = content.replace("// TEST ATTACKS HERE", "\n    ".join(perm))
    #         f.write(content_new)
    #     print("\n", perm)
    #     total, nps = time_move_gen()
    #     if nps > best_nps:
    #         best_nps = round(nps)
    #         best_order = perm
    #     print("Best nps:", best_nps)
    #     with open("src/main.c", "w") as f:
    #         f.write(content)

    # print("\nBest order:", best_order)
    # print("Best nps:", best_nps)
