from abc import ABC
import math
import os
import subprocess
import json
import sys
import time
import chess
import shutil
import pandas as pd
from tqdm import tqdm


if subprocess.run("make --version", shell=True, capture_output=True, text=True).returncode != 0:
    print("Make is not installed or not found in PATH")
    sys.exit(1)


if subprocess.run("gcc --version", shell=True, capture_output=True, text=True).returncode != 0:
    print("GCC is not installed or not found in PATH")
    sys.exit(1)


def get_position_history(board: chess.Board) -> list[str]:
    history = []
    temp_board = board.copy()

    while temp_board.move_stack:
        history.append(temp_board.fen())
        temp_board.pop()

    return history[::-1]


def get_move_history(board: chess.Board) -> list[str]:
    return [move.uci() for move in board.move_stack]


def executable(name: str) -> str:
    if sys.platform == "win32":
        return f"{name}.exe"
    else:
        return f"./{name}"


class Engine(ABC):
    def play(self, board: chess.Board, millis: int) -> dict:
        pass

    def version(self) -> str:
        return "Unknown Engine"


class SigmaZeroEngine(Engine):
    def __init__(self, exe: str = executable("engine")):
        self.make()
        self.exe = exe

    def make(self):
        result = subprocess.run("make -s", shell=True, capture_output=True)
        if result.returncode != 0:
            print("Compilation failed")
            print(result.stderr.decode())
            sys.exit(1)

    def command(self, cmd: list[str], JSON: bool = False) -> str | dict:
        try:
            result = subprocess.run(
                self.exe + " " + " ".join(f'"{arg}"' for arg in cmd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {cmd}")
            return {"error": "Command timed out"}
        if JSON:
            try:
                return dict(json.loads(result.stdout))
            except:
                return {"error": "Failed to parse JSON", "raw": result.stdout, "cmd": cmd}
        else:
            return result.stdout.strip()

    def play(self, board: chess.Board, millis: int) -> dict:
        history = get_position_history(board)
        history_str = ",".join(history) if history else ""
        if history_str:
            result = self.command(["play", board.fen(), str(millis), history_str], JSON=True)
            if result.get("error") is None:
                return result
        return self.command(["play", board.fen(), str(millis)], JSON=True)

    def version(self) -> str:
        return self.command(["version"], JSON=False)

    def moves(self, fen: str, depth: int) -> dict:
        return self.command(["moves", fen, str(depth)], JSON=True)

    def zhash(self, fen: str) -> str:
        return self.command(["hash", fen], JSON=False)

    def static_eval(self, fen: str) -> float:
        result = self.command(["eval", fen, "0"], JSON=False)
        try:
            return float(result)
        except ValueError:
            print(f"Failed to parse eval result: {result}")
            return 0.0


class OldEngine(SigmaZeroEngine):
    def __init__(self, version: str):
        if not os.path.exists(f"versions/{version}"):
            print(f"Version {version} not found in versions/")
            sys.exit(1)

        self.make(version)
        self.exe = f"versions/{version}/{executable('engine')}"

    def make(self, version: str):
        if subprocess.run("make magicbb/moves.o", shell=True, capture_output=True).returncode != 0:
            print("Compilation failed for magicbb/moves.o")
            sys.exit(1)
            
        def link_moves(filename: str):
            src_path = os.path.abspath(f"magicbb/{filename}")
            link_path = os.path.abspath(f"versions/{version}/magicbb/{filename}")
            if not os.path.exists(link_path):
                os.makedirs(os.path.dirname(link_path), exist_ok=True)
                os.symlink(src_path, link_path)
        
        link_moves("moves.o")
        link_moves("moves.c")
        
        gitignore = os.path.abspath(f"versions/{version}/magicbb/.gitignore")
        if not os.path.exists(gitignore):
            with open(gitignore, "w") as f:
                f.write("*\n")
        
        if subprocess.run(f"make -s -C versions/{version}", shell=True, capture_output=True).returncode != 0:
            print(f"Compilation failed for version {version}")
            sys.exit(1)
            
            
class UCIEngine(Engine):
    def __init__(self, exe: str):
        self.exe = exe
        self.process = subprocess.Popen(exe, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def play(self, board: chess.Board, millis: int) -> dict:
        if self.process.poll() is not None:
            print("UCI engine process has terminated unexpectedly.")
            return {"error": "Engine process terminated"}
        if not shutil.which(self.exe):
            print(f"UCI engine executable '{self.exe}' not found.")
            return {"error": "Engine executable not found"}
        
        history = get_move_history(board)
        starting_fen = get_position_history(board)[0] if history else board.fen()

        try:
            if history:
                print(f"position fen {starting_fen} moves {' '.join(history)}")
                self.process.stdin.write(f"position fen {starting_fen} moves {' '.join(history)}\n")
            else:
                self.process.stdin.write(f"position fen {board.fen()}\n")
            self.process.stdin.write(f"go movetime {millis}\n")
            self.process.stdin.flush()

            best_move = None
            depth = 0
            eval_score = 0
            time_taken = 0
            while True:
                line = self.process.stdout.readline()
                if line.startswith("bestmove"):
                    best_move = line.split()[1]
                    break
                elif line.startswith("info"):
                    parts = line.split()
                    depth = int(parts[parts.index("depth") + 1]) if "depth" in parts else depth
                    time_taken = int(parts[parts.index("time") + 1]) / 1000 if "time" in parts else time_taken
                    
                    if "cp" in parts:
                        eval_score = int(parts[parts.index("cp") + 1]) / 100
                    elif "mate" in parts:
                        eval_score = parts[parts.index("mate") + 1]

            if best_move is not None:
                return {"move": best_move, "depth": depth, "eval": eval_score, "time": time_taken}
            else:
                return {"error": "No move received from engine"}
        except Exception as e:
            print(f"Error communicating with UCI engine: {e}")
            return {"error": str(e)}

    def version(self) -> str:
        return self.exe


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


latest = SigmaZeroEngine()
old = {
    v: OldEngine(v)
    for v in os.listdir("versions")
    if os.path.isdir(os.path.join("versions", v)) and v.startswith("V2.")
}
stockfish = UCIEngine("stockfish")


if __name__ == "__main__":
    puzzle_fens = []
    with open("data/puzzles.txt", "r") as f:
        for line in f:
            fen = line.strip().split(",", 1)[0]
            if fen:
                puzzle_fens.append(fen)
    
    movegen_df = []
    for engine in sorted([latest, *old.values()], key=lambda e: e.version(), reverse=True)[:3]:
        nps_values = []
        depth_values = []
        depth = 5
        for fen in tqdm(puzzle_fens, desc=f"{engine.version()}"):
            result = engine.moves(fen, depth=depth)
            if "nps" in result:
                nps_values.append(result["nps"])
            if "depth" in result:
                depth_values.append(result["depth"])
            # if "time" in result:
            #     if result["time"] > 0.1:
            #         depth = max(1, depth - 1)
            #     elif result["time"] < 0.1:
            #         depth += 1
        
        avg_nps = sum(nps_values) / len(nps_values) if nps_values else 0
        std_nps = (sum((x - avg_nps) ** 2 for x in nps_values) / len(nps_values)) ** 0.5 if nps_values else 0
        se_nps = std_nps / len(nps_values) ** 0.5 if nps_values else 0
        avg_depth = sum(depth_values) / len(depth_values) if depth_values else 0
        movegen_df.append({
            "Engine": engine.version(),
            "Average Depth": avg_depth,
            "Average NPS": f"{avg_nps/1e6:.2f}M",
            "Uncertainty": f"±{se_nps/1e6:.2f}M"
        })
    
    movegen_df = pd.DataFrame(movegen_df)
    print(movegen_df)
    
    # TODO: Fix stockfish engine
    # tournament = Tournament(engine1=latest, engine2=stockfish, millis=(100, 100), n_games=5)
    # print(tournament.elo())
    sys.exit(0)

    # Test nps
    training_fens = []
    with open("data/training.txt", "r") as f:
        for line in f:
            fen = line.strip().split(",", 1)[0]
            if fen:
                training_fens.append(fen)

    nps = []
    depths = []
    worse_nps = ("", float("inf"))
    for fen in training_fens:
        board = chess.Board(fen)
        result = latest.play(board.fen(), 100)

        # skip results with evals that are too high, these are checkmates
        if abs(result.get("eval", 0)) > 1000:
            continue

        print(f"FEN: {fen}")
        print(f"NPS: {result.get('nps', 'N/A')}")
        print(f"Depth: {result.get('depth', 'N/A')}")
        print()
        nps.append(result.get("nps", 0))
        depths.append(result.get("depth", 0))
        if result.get("nps", float("inf")) < worse_nps[1]:
            worse_nps = (fen, result.get("nps", float("inf")))
    avg_nps = sum(nps) / len(nps)
    avg_depth = sum(depths) / len(depths)
    std_nps = (sum((x - avg_nps) ** 2 for x in nps) / len(nps)) ** 0.5
    std_depth = (sum((x - avg_depth) ** 2 for x in depths) / len(depths)) ** 0.5
    se_nps = std_nps / len(nps) ** 0.5
    se_depth = std_depth / len(depths) ** 0.5
    print(f"Average NPS: {avg_nps:,.0f} ± {se_nps:,.0f}")
    print(f"Average Depth: {avg_depth:.2f} ± {se_depth:.2f}")
    print(f"Worst NPS: {worse_nps[1]:.0f} for FEN:\n{worse_nps[0]}")
