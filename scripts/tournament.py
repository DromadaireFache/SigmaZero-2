import math
import sys
import time
import chess

from .engines import Engine
from . import chessdata


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
    yield from chessdata.Dataloader(n=n, postype=chessdata.PosType.OPENING, balanced=True, repeat=True).fens()


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
        self.elo_is_available = False
        self.results = self.run()

    def play_game(self, fen: str, is_white: bool) -> dict:
        results = {"score": 0, "time_2": 0, "time_1": 0, "avg_depth_1": 0, "avg_depth_2": 0}
        board = chess.Board(fen)
        number_of_moves = 0

        if self.score_to_beat is not None:
            print(self.engine1.version(), "v", self.engine2.version(), f"({self.score_to_beat})")
        else:
            print(self.engine1.version(), "v", self.engine2.version())

        while not board.is_game_over(claim_draw=True) and number_of_moves < 150:
            if (board.turn == chess.WHITE and is_white) or (board.turn == chess.BLACK and not is_white):
                result = self.engine1.play(board, self.millis[0])
                results["time_1"] += result.get("time", 0)
                results["avg_depth_1"] += result.get("depth", 0)
            else:
                result = self.engine2.play(board, self.millis[1])
                results["time_2"] += result.get("time", 0)
                results["avg_depth_2"] += result.get("depth", 0)

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
            results["avg_depth_1"] /= number_of_moves / 2
            results["avg_depth_2"] /= number_of_moves / 2

        return results

    def run(self) -> dict:
        start = time.perf_counter()
        results = {"wins": 0, "losses": 0, "draws": 0, "avg_depth_1": 0, "avg_depth_2": 0}
        for i, fen in enumerate(get_tournament_fens(self.n_games)):
            try:
                if i == 0:
                    print(f"Game {i+1}/{self.n_games}")
                else:
                    time_left = (time.perf_counter() - start) * (self.n_games - i) / i
                    print(f"Game {i+1}/{self.n_games} (time left: {time_left:.0f}s)")
                result = self.play_game(fen, is_white=(i % 2 == 0))
                print("End FEN:", result.get("end_fen", "N/A"))
                print(f"Time {result['time_1']:.2f}s / {result['time_2']:.2f}s")
                print(f"Avg Depth {result['avg_depth_1']:.2f} / {result['avg_depth_2']:.2f}")

                results["avg_depth_1"] *= i / (i + 1)
                results["avg_depth_2"] *= i / (i + 1)
                results["avg_depth_1"] += result["avg_depth_1"] / (i + 1)
                results["avg_depth_2"] += result["avg_depth_2"] / (i + 1)

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

        print(f"Tournament completed in {time.perf_counter() - start:.2f} seconds.")
        print("Tournament Results:")
        print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")

        # Calculate the elo uncertainty and 95% confidence interval
        n = results["wins"] + results["losses"] + results["draws"]
        w, l, d = results["wins"], results["losses"], results["draws"]
        if n > 0 and w + d > 0:  # Avoid division by zero
            self.elo_is_available = True
            s = (w + 0.5 * d) / n
            elo = 400 * math.log10(s / (1 - s))
            # trinomial variance of the score
            sigma_s = (w / n * (1 - s) ** 2 + l / n * s**2 + d / n * (0.5 - s) ** 2) ** 0.5 / n
            sigma_elo = 400 / math.log(10) * sigma_s / (s * (1 - s))
            ci_95 = 1.96 * sigma_elo
            print(f"Elo: {elo:+.2f} ± {ci_95:.2f} (95% confidence interval)")
            result["elo"] = elo
            result["elo_ci"] = ci_95

        depth_diff = results["avg_depth_1"] - results["avg_depth_2"]
        depth_diff = ("+" if depth_diff >= 0 else "") + f"{depth_diff:.2f}"
        print(
            f"Average Depth {self.engine1.version()}: {results['avg_depth_1']:.2f}, "
            f"{self.engine2.version()}: {results['avg_depth_2']:.2f} ({depth_diff})"
        )

        return results

    def elo(self) -> float:
        if not self.elo_is_available:
            raise ValueError("Elo is not available for this tournament. Not enough games played.")
        return self.results.get("elo", 0.0)

    def score(self) -> tuple[bool, int]:
        score = self.results.get("wins", 0) - self.results.get("losses", 0)
        return score > self.required_score, score
