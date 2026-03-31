import math
import sys
import time
import chess

from .engines import Engine


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


def get_tournament_fens(n: int = None):
    seen_fens = set()
    with open("data/chessData.csv", "r") as f:
        i = 0
        for line_index, line in enumerate(f):
            if line_index % 10 != 0:
                continue # Only take every 10th line to get more variety and avoid similar positions
            if n is not None and i >= n:
                break
            if not line.strip():
                continue

            fen, score = line.strip().split(",", 1)

            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            try:
                score = int(score)
                fullmoves = int(fen.split()[-1])
            except ValueError:
                continue

            if fullmoves < 10 and abs(score) < 40:
                yield fen
                i += 1
                if i < n:
                    yield fen  # Twice to play both colors
                    i += 1


class Tournament:
    def __init__(
        self,
        engine1: Engine,
        engine2: Engine,
        millis: int | tuple[int, int],
        n_games: int,
        score_to_beat: int | None = None,
        required_score: int = -100,
        exit_on_interrupt: bool = False,
    ):
        self.engine1 = engine1
        self.engine2 = engine2
        self.n_games = n_games
        if isinstance(millis, int):
            self.millis = (millis, millis)
        else:
            self.millis = millis
        self.score_to_beat = score_to_beat
        self.required_score = required_score
        self.exit_on_interrupt = exit_on_interrupt
        self.results = self.run()

    def play_game(self, fen: str, is_white: bool) -> dict:
        results = {"score": 0, "time_old": 0, "time_new": 0, "avg_depth_new": 0, "avg_depth_old": 0}
        board = chess.Board(fen)
        number_of_moves = 0

        if self.score_to_beat is not None:
            print(self.engine1.version(), "v", self.engine2.version(), f"({self.score_to_beat})")
        else:
            print(self.engine1.version(), "v", self.engine2.version())

        while not board.is_game_over(claim_draw=True) and number_of_moves < 150:
            if (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and not is_white):
                result = self.engine1.play(board, self.millis[0])
                results["time_new"] += result.get("time", 0)
                results["avg_depth_new"] += result.get("depth", 0)
            else:
                result = self.engine2.play(board, self.millis[1])
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
        elif number_of_moves >= 150:
            print("Game drawn by move limit")
        else:
            what_draw(board)

        results["end_fen"] = board.fen()
        if number_of_moves > 0:
            results["avg_depth_new"] /= number_of_moves / 2
            results["avg_depth_old"] /= number_of_moves / 2

        return results

    def run(self) -> dict:
        start = time.perf_counter()
        results = {"wins": 0, "losses": 0, "draws": 0, "avg_depth_new": 0, "avg_depth_old": 0}
        for i, fen in enumerate(get_tournament_fens(self.n_games)):
            try:
                print(f"Game {i+1}/{self.n_games}")
                result = self.play_game(fen, is_white=(i % 2 == 0))
                print("End FEN:", result.get("end_fen", "N/A"))
                print(f"Time SigmaZero: {result['time_new']:.2f}s, Old: {result['time_old']:.2f}s")
                print(f"Avg Depth SigmaZero: {result['avg_depth_new']:.2f}, Old: {result['avg_depth_old']:.2f}")

                results["avg_depth_new"] *= i / (i + 1)
                results["avg_depth_old"] *= i / (i + 1)
                results["avg_depth_new"] += result["avg_depth_new"] / (i + 1)
                results["avg_depth_old"] += result["avg_depth_old"] / (i + 1)

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

                best_possible_score = self.n_games - i + results["wins"] - results["losses"]
                if best_possible_score <= self.required_score:
                    print("Stopping tournament, impossible to achieve required score at this point.")
                    return results
            except KeyboardInterrupt as e:
                if self.exit_on_interrupt:
                    print("Tournament interrupted.")
                    break
                else:
                    raise e

        # Calculate Elo difference. Add a small pseudocount for sweep cases to avoid division by zero.
        score_for = results["wins"] + 0.5 * results["draws"]
        score_against = results["losses"] + 0.5 * results["draws"]
        if score_for == 0 and score_against == 0:
            elo = 0.0
        elif score_for == 0 or score_against == 0:
            elo = 400 * math.log10((score_for + 0.5) / (score_against + 0.5))
        else:
            elo = 400 * math.log10(score_for / score_against)
        results["elo"] = elo

        print(f"Tournament completed in {time.perf_counter() - start:.2f} seconds.")
        print("Tournament Results:")
        print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}, {elo:+.2f} Elo")

        depth_diff = results["avg_depth_new"] - results["avg_depth_old"]
        depth_diff = ("+" if depth_diff >= 0 else "") + f"{depth_diff:.2f}"
        print(
            f"Average Depth SigmaZero: {results['avg_depth_new']:.2f}, Old: {results['avg_depth_old']:.2f} ({depth_diff})"
        )

        return results

    def elo(self) -> float:
        return self.results.get("elo", 0.0)

    def score(self) -> tuple[bool, int]:
        score = self.results.get("wins", 0) - self.results.get("losses", 0)
        return score > self.required_score, score
