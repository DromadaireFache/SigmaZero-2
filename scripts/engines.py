from abc import ABC
import json
import os
import shutil
import subprocess
import sys

import chess


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

    history.append(temp_board.fen())  # Add the initial position
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
