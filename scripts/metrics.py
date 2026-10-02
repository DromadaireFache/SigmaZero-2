from time import time
from tqdm import TqdmWarning, tqdm
import warnings

from .engines import SigmaZeroEngine, UCIEngine
from . import chessdata
from .tournament import Tournament

warnings.filterwarnings("ignore", category=TqdmWarning)


def mean_and_se(values: list[float | int]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if values else 0
    se = std / len(values) ** 0.5 if values else 0
    return mean, se


class TimedProgress:
    def __init__(self, name: str, duration: int):
        self.end = time() + duration
        self.pbar = tqdm(total=duration, desc=name, unit="s", leave=False)
        self._last = time()

    def __bool__(self):
        now = time()
        self.pbar.update(now - self._last)
        self._last = now
        remaining = self.end - now
        if remaining <= 0:
            self.pbar.close()
            return False
        return True


def print_formatted_results(results: dict[str, int | float | tuple[float, float]]):
    print("\nMetric Results:")
    for metric, result in results.items():
        if isinstance(result, tuple):
            mean, se = result
            if metric.lower() == "time":
                print(f"{metric}: {mean:.2f}s ± {se:.2f}s")
            elif mean >= 1e6:
                print(f"{metric}: {mean/1e6:.2f}M ± {se/1e6:.2f}M")
            else:
                print(f"{metric}: {mean:.2f} ± {se:.2f}")
                
        else:
            if metric.lower() == "time":
                print(f"{metric}: {result:.2f}s")
            elif result >= 1e6:
                print(f"{metric}: {result/1e6:.2f}M")
            elif isinstance(result, float):
                print(f"{metric}: {result:.2f}")
            else:
                print(f"{metric}: {result}")


def report_movegen(engine: SigmaZeroEngine, duration: int = 60):
    times = []
    depths = []
    nps_values = []
    depth = 5
    fens = chessdata.Dataloader().fens()
    progress = TimedProgress("movegen", duration)

    while progress:
        fen = next(fens)
        result = engine.moves(fen, depth=depth)

        if "time" in result:
            times.append(result["time"])
        if "depth" in result:
            depths.append(result["depth"])
        if "nps" in result:
            nps_values.append(result["nps"])

        # Want to balance the depth to get ~0.5s per position on average
        avg_time = sum(times) / len(times) if times else 0.5
        if avg_time > 1.0:
            depth = max(1, depth - 1)
        elif avg_time < 0.02:
            depth = min(6, depth + 1)

    print_formatted_results({"Time": mean_and_se(times), "Depth": mean_and_se(depths), "NPS": mean_and_se(nps_values)})


def report_depth(engine: SigmaZeroEngine, duration: int = 60):
    engine.untrack_metrics()
    engine.make()
    depths = []
    depths_endgame = []
    depths_earlygame = []
    fens = chessdata.Dataloader().fens()
    progress = TimedProgress("depth", duration)

    while progress:
        fen = next(fens)
        postype = chessdata.get_postype(fen)
        result = engine.play(fen, 100)

        if "depth" in result and result["depth"] > 0:
            depths.append(result["depth"])
            if postype == chessdata.PosType.ENDGAME:
                depths_endgame.append(result["depth"])
            else:
                depths_earlygame.append(result["depth"])

    print_formatted_results(
        {
            "Depth": mean_and_se(depths),
            "Depth (Endgame)": mean_and_se(depths_endgame),
            "Depth (Earlygame)": mean_and_se(depths_earlygame),
        }
    )


def report_accuracy(engine: SigmaZeroEngine, duration: int = 60):
    engine.untrack_metrics()
    engine.make()
    accuracies_earlygame = []
    accuracies_endgame = []
    fens = chessdata.Dataloader().fen_moves()  # Get FENs with moves for accuracy testing
    progress = TimedProgress("accuracy", duration)

    while progress:
        fen, move = next(fens)
        result = engine.play(fen, 100)

        if "move" in result:
            predicted_move = result["move"]
            postype = chessdata.get_postype(fen)
            is_correct = (predicted_move == move) * 100  # Scale to percentage
            if postype == chessdata.PosType.ENDGAME:
                accuracies_endgame.append(is_correct)
            else:
                accuracies_earlygame.append(is_correct)

    accuracies = accuracies_earlygame + accuracies_endgame
    print_formatted_results(
        {
            "Accuracy": mean_and_se(accuracies),
            "Accuracy (Endgame)": mean_and_se(accuracies_endgame),
            "Accuracy (Earlygame)": mean_and_se(accuracies_earlygame),
        }
    )


def report_checkmates(engine: SigmaZeroEngine, duration: int = 60):
    """
    Reports the number of moves the engine took to checkmate stockfish in winning endgames.
    This function purely exists to fix the bug where the engine sometimes fails to find a checkmate in winning endgames.
    """
    fens = [
        "8/8/3k4/8/8/3KR3/8/8 w - - 0 60",  # KR vs k
        "8/8/3kr3/8/8/3K4/8/8 b - - 0 60",  # K vs kr
        # "8/8/3k4/8/8/3KQ3/8/8 w - - 0 60",  # KQ vs k
        # "8/8/3kq3/8/8/3K4/8/8 b - - 0 60",  # K vs kq
        # "8/8/3k4/8/8/3KP3/8/8 w - - 0 60",  # KP vs k
        # "8/8/3kp3/8/8/3K4/8/8 b - - 0 60",  # K vs kp
        # "8/8/3k4/8/8/3KBB2/8/8 w - - 0 60",  # KBB vs k
        # "8/8/3kbb2/8/8/3K4/8/8 b - - 0 60",  # K vs kbb
        # "8/8/3k4/8/8/3KNN2/8/8 w - - 0 60",  # KNN vs k
        # "8/8/3knn2/8/8/3K4/8/8 b - - 0 60",  # K vs knn
        # "8/8/3k4/8/8/3KNB2/8/8 w - - 0 60",  # KNB vs k
        # "8/8/3knb2/8/8/3K4/8/8 b - - 0 60",  # K vs knb
    ]

    n_draws = 0
    n_wins = 0
    moves_to_checkmate = []

    tournament = Tournament(
        engine1=engine,
        engine2=UCIEngine("stockfish"),
        millis=(100, 10), # Give stockfish 10ms per move to make it play instantly
        n_games=0, # We supply the games one by one
        auto_run=False,
    )
    
    for fen in fens:
        for _ in range(20): # Play each position 5 times to account for randomness
            results = tournament.play_game(fen, is_white=fen.split()[1] == "w")
            if results["score"] == 1:
                n_wins += 1
                moves_to_checkmate.append(results["n_moves"])
                print(f"{fen.ljust(40)}  Checkmate in {results['n_moves']} moves")
            elif results["score"] == 0:
                n_draws += 1
                print(f"{fen.ljust(40)}  Draw")
            else:
                print("Something went wrong with the game.")
            print("")
            
    print_formatted_results(
        {
            "Checkmates Found": n_wins,
            "Draws": n_draws,
            "Moves to Checkmate": mean_and_se(moves_to_checkmate),
        }
    )


available_metrics = {
    "movegen": report_movegen,
    "depth": report_depth,  # Placeholder for depth metric
    "nodes": lambda engine, duration: None,  # Placeholder for nodes metric
    "beta_cutoffs": lambda engine, duration: None,  # Placeholder for beta cutoffs metric
    "accuracy": report_accuracy,
    "checkmates": report_checkmates,
}


def report(engine: SigmaZeroEngine, metrics_to_track: list[str], duration: int = 60):
    duration_per_metric = max(1, duration // len(metrics_to_track))
    for metric in metrics_to_track:
        available_metrics[metric](engine, duration=duration_per_metric)
