#!/usr/bin/env python3
import os
import sys
import chess
import pandas as pd
from tqdm import tqdm
import argparse
import shutil

from scripts.engines import UCIEngine, OldEngine, SigmaZeroEngine
from scripts.tournament import Tournament
from scripts import metrics
import app.main as app


latest = SigmaZeroEngine()
old = {v: OldEngine(v) for v in os.listdir("versions") if os.path.isdir(os.path.join("versions", v))}
stockfish = UCIEngine("stockfish")
engines = {"latest": latest, "stockfish": stockfish, **old}


def get_engines(arg1, arg2):
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
    p.add_argument("--millis", type=int, nargs="+", default=[10], help="Time per move in ms (default: 10)")
    p.add_argument("--games", type=int, default=500, help="Number of games to play (default: 500)")

    # Subcommand for running the app
    p = subparsers.add_parser("app", help="Run the pywebview app")

    # Subcommand for archiving a new version of SigmaZero
    p = subparsers.add_parser("archive", help="Archive a new version of SigmaZero")
    p.add_argument("version", type=str, help="Version name for the new archive (e.g., V2.1)")

    # Subcommand for metric tracking
    p = subparsers.add_parser("metrics", help="Track specific metrics in the engine code")
    p.add_argument("engine", type=str, default="latest", help="Engine to track metrics for (default: latest)")
    p.add_argument("metrics", type=str, nargs="+", choices=metrics.available_metrics, help=f"List of metrics to track")
    p.add_argument("--duration", type=int, default=60, help="Duration to track metrics in seconds (default: 60)")

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

    elif args.command == "archive":
        version: str = args.version
        if version in old or version == "latest":
            parser.error(f"Version '{version}' already exists. Please choose a different version name.")
        archive_path = os.path.join("versions", version)
        os.makedirs(archive_path, exist_ok=False)
        shutil.copytree("src", os.path.join(archive_path, "src"))
        shutil.copy("Makefile", os.path.join(archive_path, "Makefile"))

    elif args.command == "metrics":
        engine = engines.get(args.engine)
        if not isinstance(engine, SigmaZeroEngine):
            parser.error("Metrics tracking is only supported for SigmaZero engine.")
        metrics.report(engine, args.metrics, duration=args.duration)

    else:
        print("No command specified. Use --help for available commands.")
        sys.exit(1)
