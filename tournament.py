import sys
import time
from src import sigma_zero
import chess

TIME = 100  # milliseconds per move
OLD_BOT = "V2.0"
FENS = []

with open("data/FENs.txt", "r") as f:
    for line in f:
        fen = line.strip()
        if fen:  # append twice for both colors
            FENS.append(fen)
            FENS.append(fen)

sigma_zero.make()
sigma_zero.make(OLD_BOT)


def what_draw(board: chess.Board) -> str:
    if board.is_stalemate():
        print("Draw by stalemate")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.is_seventyfive_moves():
        print("Draw by 75-move rule")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition")
    elif board.is_fifty_moves():
        print("Draw by 50-move rule")
    elif board.can_claim_threefold_repetition():
        print("Draw by threefold repetition")
    else:
        print("Draw by agreement or unknown reason")


def illegal_move(board: chess.Board, move_uci: str, result: dict):
    print("Bot suggested an illegal move. Exiting.")
    print(f"{result=}")
    print(f"Board FEN:\n{board.fen()}")
    print(f"Illegal move UCI: {move_uci}")
    sys.exit(1)


def play_game(fen: str, is_white: bool) -> dict:
    results = {"score": 0, "time_old": 0, "time_new": 0, "avg_depth_new": 0, "avg_depth_old": 0}
    board = chess.Board(fen)
    number_of_moves = 0
    while not board.is_game_over(claim_draw=True):
        if (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and not is_white):
            result = sigma_zero.play(board.fen(), TIME)
            results["time_new"] += result.get("time", 0)
            results["avg_depth_new"] += result.get("depth", 0)
        else:
            result = sigma_zero.old_play(board.fen(), TIME)
            results["time_old"] += result.get("time", 0)
            results["avg_depth_old"] += result.get("depth", 0)
        move_uci = result.get("move", "<unknown>")
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                number_of_moves += 1
            else:
                illegal_move(board, move_uci, result)
        except Exception:
            illegal_move(board, move_uci, result)

    if board.result(claim_draw=True) == "1-0":
        results["score"] = 1 if is_white else -1
    elif board.result(claim_draw=True) == "0-1":
        results["score"] = -1 if is_white else 1
    else:
        what_draw(board)
    results["end_fen"] = board.fen()
    if number_of_moves > 0:
        results["avg_depth_new"] /= number_of_moves / 2
        results["avg_depth_old"] /= number_of_moves / 2
    return results


def tournament():
    start = time.perf_counter()
    results = {"wins": 0, "losses": 0, "draws": 0}
    for i, fen in enumerate(FENS):
        try:
            print(f"Game {i+1}/{len(FENS)}")
            result = play_game(fen, is_white=(i % 2 == 0))
            print("End FEN:", result.get("end_fen", "N/A"))
            print(f"Time SigmaZero: {result['time_new']:.2f}s, Old: {result['time_old']:.2f}s")
            print(f"Avg Depth SigmaZero: {result['avg_depth_new']:.2f}, Old: {result['avg_depth_old']:.2f}")
            if result["score"] == 1:
                results["wins"] += 1
                print("Result: Win", end=" ")
            elif result["score"] == -1:
                results["losses"] += 1
                print("Result: Loss", end=" ")
            else:
                results["draws"] += 1
                print("Result: Draw", end=" ")
            print(f"({results['wins']}W/{results['losses']}L/{results['draws']}D)\n")
        except KeyboardInterrupt:
            print("Tournament interrupted.")
            break
    print(f"Tournament completed in {time.perf_counter() - start:.2f} seconds.")
    print("Tournament Results:")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")


if __name__ == "__main__":
    tournament()
