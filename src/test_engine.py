from pprint import pprint
import sys
import chess
import pytest

try:
    import sigma_zero
except ModuleNotFoundError:
    from . import sigma_zero

DEPTH = 4


def count_moves(board: chess.Board, depth: int):
    if depth == 0:
        return 1
    elif depth == 1:
        return board.legal_moves.count()
    n = 0

    # if depth == 1:
    #     mybot = sigma_zero.moves(board.fen(), 1).get("nodes", -1)
    #     if mybot != board.legal_moves.count():
    #         print(mybot, board.legal_moves.count())
    #         print(f"Invalid number of moves:\n{board.fen()}")
    #         sys.exit(1)
    #     return board.legal_moves.count()

    for move in board.legal_moves:
        board.push(move)
        x = count_moves(board, depth - 1)
        # if depth == DEPTH:
        #     print(f"{move}: {x}")
        n += x
        board.pop()
    return n


@pytest.mark.parametrize("fen", sigma_zero.FENS)
def test_move_generation(fen: str):
    mybot = sigma_zero.moves(fen, DEPTH).get("nodes", -1)
    chesslib = count_moves(chess.Board(fen), DEPTH)
    assert mybot == chesslib


def find_wrong_move_generation(board: chess.Board, depth: int, search_depth: int = 1):
    chesslib_moves = count_moves(board, search_depth)
    n_moves = sigma_zero.moves(board.fen(), search_depth).get("nodes", -1)
    if chesslib_moves != n_moves:
        print(f"Mismatch found at depth {depth} for FEN:\n{board.fen()}")
        print(f"Chess lib moves: {chesslib_moves}, SigmaZero moves: {n_moves}")
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

    FEN = "r4k2/p4prp/2qb4/8/2pP4/2P5/PP1N1QPP/R3R1K1 b - - 0 1"
    DEPTH = 5

    print("----- Chess lib -----")
    print("total =", count_moves(chess.Board(FEN), DEPTH))
    print("\n----- SigmaZero 2 -----")
    print("mybot = ", sigma_zero.moves(FEN, DEPTH).get("nodes", -1))

    # find_wrong_move_generation(chess.Board(FEN), 1, DEPTH - 1)
    # print("No mismatches found")

    # board = chess.Board(FEN)
    # for move in board.legal_moves:
    #     board.push(move)
    #     fen = board.fen()
    #     mybot = sigma_zero.moves(fen, DEPTH - 1)
    #     if not mybot:
    #         print(fen)
    #         print(move)
    #     board.pop()
