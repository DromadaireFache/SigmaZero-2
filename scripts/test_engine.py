import os
import sys
import chess
import pytest
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sigma_zero
from scripts import chessdata

DEPTH = 5
N_POSITIONS = 100

def count_moves(board: chess.Board, depth: int):
    """Count moves using Stockfish perft."""
    return sigma_zero.stockfish.moves(board.fen(), depth).get("nodes", -1)


@pytest.mark.parametrize("fen", chessdata.Dataloader(n=N_POSITIONS).fens())
def test_move_generation(fen: str):
    mybot = sigma_zero.latest.moves(fen, DEPTH).get("nodes", -1)
    chesslib = count_moves(chess.Board(fen), DEPTH)
    assert mybot == chesslib


def find_wrong_move_generation(board: chess.Board, depth: int, search_depth: int = 1):
    sf_moves = count_moves(board, search_depth)
    n_moves = sigma_zero.latest.moves(board.fen(), search_depth).get("nodes", -1)
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
    print("mybot = ", sigma_zero.latest.moves(FEN, DEPTH).get("nodes", -1))

    find_wrong_move_generation(chess.Board(FEN), DEPTH)
    print("No mismatches found")
    
    # fens = chessdata.Dataloader().fens()
    # for fen in tqdm(fens):
    #     a = set(sigma_zero.latest.moves(fen, 0)["moves"])
    #     b = set(sigma_zero.old["V2.9"].moves(fen, 0)["moves"])
    #     if a != b:
    #         print(f"Mismatch found for FEN:\n{fen}")
    #         print(f"SigmaZero 2 moves: {a}")
    #         print(f"SigmaZero 1 moves: {b}")
    #         if len(a) < len(b):
    #             print("Missing moves:", b - a)
    #         else:
    #             print("Extra moves:", a - b)
    #         sys.exit(1)
            
        # a = sigma_zero.latest.moves(fen, 3)["nodes"]
        # b = sigma_zero.old["V2.9"].moves(fen, 3)["nodes"]
        # if a != b:
        #     print(f"Mismatch found for FEN:\n{fen}")
        #     print(f"SigmaZero 2 nodes: {a}")
        #     print(f"SigmaZero 1 nodes: {b}")
        #     if a < b:
        #         print("Missing nodes:", b - a)
        #     else:
        #         print("Extra nodes:", a - b)
        #     sys.exit(1)

    # board = chess.Board(FEN)
    # for move in board.legal_moves:
    #     board.push(move)
    #     fen = board.fen()
    #     mybot = sigma_zero.latest.moves(fen, DEPTH - 1)
    #     if not mybot:
    #         print(fen)
    #         print(move)
    #     board.pop()
