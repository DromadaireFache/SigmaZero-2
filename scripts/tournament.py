from dataclasses import dataclass
import math
import sys
import time
import chess
from func_timeout import func_set_timeout, FunctionTimedOut

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

    @func_set_timeout(300)  # 5 minute timeout for a single game to prevent hanging
    def play_game(self, fen: str, is_white: bool) -> dict:
        results = {"score": 0, "time_2": 0, "time_1": 0, "avg_depth_1": 0, "avg_depth_2": 0}
        board = chess.Board(fen)
        number_of_moves = 0

        if hasattr(self, "score_to_beat"):
            if self.score_to_beat is not None:
                print(self.engine1.version(), "v", self.engine2.version(), f"({self.score_to_beat})")
            else:
                print(self.engine1.version(), "v", self.engine2.version())

        def move_number_limit_reached(n: int, eval: float) -> bool:
            if n < 150:
                return False
            return abs(eval) < 5.0 or n >= 250  # Only stop if an engine is not winning by more than a queen

        while not board.is_game_over(claim_draw=True):
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

            if move_number_limit_reached(number_of_moves, result.get("eval", 0)):
                break

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
            sigma_s = math.sqrt(w * (1 - s) ** 2 + l * s**2 + d * (0.5 - s) ** 2) / n
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


@dataclass
class SprtResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    verdict: str = "inconclusive"  # "accept", "reject", "inconclusive"
    elo: float = 0.0
    elo_ci: float = 0.0
    llr: float = 0.0
    avg_depth_1: float = 0.0
    avg_depth_2: float = 0.0


def _llr(w, l, d, elo0, elo1):
    """
    Log-likelihood ratio for trinomial (W/D/L) outcomes.
    Tests H0: true Elo = elo0  vs  H1: true Elo = elo1.
    """
    n = w + l + d
    if n == 0:
        return 0.0

    s = (w + 0.5 * d) / n

    def expected_score(elo_diff):
        return 1 / (1 + 10 ** (-elo_diff / 400))

    def log_likelihood(elo_diff):
        e = expected_score(elo_diff)
        # Trinomial probabilities: wins ~ e^2, draws ~ 2e(1-e), losses ~ (1-e)^2
        # (Simplified BayesElo model — good enough for engine testing)
        pw = e * e
        pd = 2 * e * (1 - e)
        pl = (1 - e) * (1 - e)
        # Clamp to avoid log(0)
        pw, pd, pl = max(pw, 1e-10), max(pd, 1e-10), max(pl, 1e-10)
        return w * math.log(pw) + d * math.log(pd) + l * math.log(pl)

    return log_likelihood(elo1) - log_likelihood(elo0)


def _elo_and_ci(w, l, d):
    n = w + l + d
    if n == 0 or w + d == 0:
        return 0.0, float("inf")
    s = (w + 0.5 * d) / n
    if s <= 0 or s >= 1:
        return 0.0, float("inf")
    elo = 400 * math.log10(s / (1 - s))
    sigma_s = math.sqrt(w * (1 - s) ** 2 + l * s**2 + d * (0.5 - s) ** 2) / n
    sigma_elo = 400 / math.log(10) * sigma_s / (s * (1 - s))
    return elo, 1.96 * sigma_elo


class SprtTournament(Tournament):
    def __init__(
        self,
        engine1: Engine,
        engine2: Engine,
        millis: int | tuple[int, int],
        elo0: float = 0.0,  # H0: not worth shipping (below this)
        elo1: float = 10.0,  # H1: worth shipping (at least this)
        alpha: float = 0.05,  # false positive rate
        beta: float = 0.05,  # false negative rate
        max_games: int = 1000,  # safety cap — at 10ms ≈ 30 min
        exit_on_interrupt: bool = False,
    ):
        self.engine1 = engine1
        self.engine2 = engine2
        self.millis = (millis, millis) if isinstance(millis, int) else millis
        self.elo0 = elo0
        self.elo1 = elo1
        self.exit_on_interrupt = exit_on_interrupt
        self.max_games = max_games
        self.alpha = alpha
        self.beta = beta

        # SPRT bounds (Wald's sequential test)
        self.upper = math.log((1 - beta) / alpha)  # accept H1
        self.lower = math.log(beta / (1 - alpha))  # accept H0 (reject change)

        self.result = self.run()

    def run(self) -> SprtResult:
        res = SprtResult()
        start = time.perf_counter()
        games_played = 0

        print(f"SPRT bounds: lower={self.lower:.3f}, upper={self.upper:.3f}")
        print(f"H0={self.elo0} Elo, H1={self.elo1} Elo, α={self.alpha}, β={self.beta}\n")

        for i, fen in enumerate(get_tournament_fens(self.max_games)):
            if games_played >= self.max_games:
                break
            try:
                # Play both colors from this opening (one pair = two games)
                for is_white in (True, False):
                    if games_played >= self.max_games:
                        break

                    elapsed = time.perf_counter() - start
                    if games_played > 0:
                        eta = elapsed * (self.max_games - games_played) / games_played
                        print(
                            f"Game {games_played+1}/{self.max_games}  "
                            f"LLR={res.llr:+.3f} [{self.lower:.2f}, {self.upper:.2f}]  "
                            f"ETA {eta:.0f}s"
                        )

                    game = self.play_game(fen, is_white)
                    games_played += 1

                    if game["score"] == 1:
                        res.wins += 1
                        label = "Win"
                    elif game["score"] == -1:
                        res.losses += 1
                        label = "Loss"
                    else:
                        res.draws += 1
                        label = "Draw"

                    # Running average depth
                    n = games_played
                    res.avg_depth_1 = res.avg_depth_1 * (n - 1) / n + game["avg_depth_1"] / n
                    res.avg_depth_2 = res.avg_depth_2 * (n - 1) / n + game["avg_depth_2"] / n

                    w, l, d = res.wins, res.losses, res.draws
                    res.llr = _llr(w, l, d, self.elo0, self.elo1)
                    res.elo, res.elo_ci = _elo_and_ci(w, l, d)

                    print(
                        f"  {label}  ({w}W/{l}L/{d}D)  "
                        f"Elo {res.elo:+.1f} ± {res.elo_ci:.1f}  "
                        f"LLR {res.llr:+.3f}"
                    )

                    if res.llr >= self.upper:
                        res.verdict = "accept"
                        self._print_summary(res, start, games_played)
                        return res
                    if res.llr <= self.lower:
                        res.verdict = "reject"
                        self._print_summary(res, start, games_played)
                        return res

            except KeyboardInterrupt:
                if self.exit_on_interrupt:
                    print("Interrupted.")
                    break
                raise

        res.verdict = "inconclusive"
        self._print_summary(res, start, games_played)
        return res

    def _print_summary(self, res: SprtResult, start: float, n: int):
        elapsed = time.perf_counter() - start
        print(f"\n{'='*50}")
        print(f"Verdict: {res.verdict.upper()}  ({n} games in {elapsed:.0f}s)")
        print(f"W={res.wins} L={res.losses} D={res.draws}")
        print(f"Elo: {res.elo:+.2f} ± {res.elo_ci:.2f}  LLR: {res.llr:+.3f}")
        print(f"Avg depth: {res.avg_depth_1:.1f} vs {res.avg_depth_2:.1f}")
        print(f"{'='*50}\n")
