#!/usr/bin/env python3
import os
import sys
import chess
import pandas as pd
from tqdm import tqdm
import argparse

from scripts.engines import UCIEngine, OldEngine, SigmaZeroEngine
from scripts.tournament import Tournament
import app.main as app


latest = SigmaZeroEngine()
old = {
    v: OldEngine(v)
    for v in os.listdir("versions")
    if os.path.isdir(os.path.join("versions", v)) and v.startswith("V2.")
}
stockfish = UCIEngine("stockfish")


def get_engines(arg1, arg2):
    engines = {"latest": latest, "stockfish": stockfish, **old}
    engine1 = engines.get(arg1)
    engine2 = engines.get(arg2)
    if engine1 is None or engine2 is None:
        print(f"Invalid engines specified. Available engines: {', '.join(engines.keys())}")
        sys.exit(1)
    return engine1, engine2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for running tournaments
    p = subparsers.add_parser("tournament", help="Run a tournament between engines")
    p.add_argument("engine1", type=str, default="latest", help="First engine to compete (default: latest)")
    p.add_argument("engine2", type=str, default="stockfish", help="Second engine to compete (default: stockfish)")
    p.add_argument("--millis", type=int, nargs="+", default=[100], help="Time per move in ms (default: 100)")
    p.add_argument("--games", type=int, default=100, help="Number of games to play (default: 100)")
    
    # Subcommand for running the app
    p = subparsers.add_parser("app", help="Run the pywebview app")

    args = parser.parse_args()

    if args.command == "tournament":
        engine1, engine2 = get_engines(args.engine1, args.engine2)
        if len(args.millis) > 2:
            parser.error("Too many time controls specified. Provide at most two values for --millis.")
        elif len(args.millis) == 1:
            millis = (args.millis[0], args.millis[0])
        else:
            millis = (args.millis[0], args.millis[1])
        Tournament(engine1=engine1, engine2=engine2, millis=millis, n_games=args.games, exit_on_interrupt=True)
        
    elif args.command == "app":
        app.start("app/index.html")

    else:
        print("No command specified. Use --help for available commands.")
        sys.exit(1)


def random_stuff_i_need_to_move_somewhere_else():
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
        movegen_df.append(
            {
                "Engine": engine.version(),
                "Average Depth": avg_depth,
                "Average NPS": f"{avg_nps/1e6:.2f}M",
                "Uncertainty": f"±{se_nps/1e6:.2f}M",
            }
        )

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
