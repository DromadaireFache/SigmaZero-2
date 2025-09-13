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


if __name__ == "__main__":
    pprint(sigma_zero.old_play("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 10000))
    pprint(sigma_zero.play("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2000))


# print("----- Chess lib -----")
# print("total =", count_moves(chess.Board(FEN), DEPTH))
# print("\n----- SigmaZero 2 -----")
# sigma_zero.moves(FEN, DEPTH)

# board = chess.Board(FEN)
# for move in board.legal_moves:
#     board.push(move)
#     fen = board.fen()
#     mybot = sigma_zero.moves(fen, DEPTH - 1)
#     if not mybot:
#         print(fen)
#         print(move)
#     board.pop()
