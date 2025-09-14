from dataclasses import dataclass
from src import sigma_zero

TIME = 1000  # milliseconds per move


@dataclass
class Puzzle:
    fen: str
    move: str


with open("data/puzzles.txt", "r") as f:
    PUZZLES = [Puzzle(*line.strip().split(",")) for line in f if line.strip()]

sigma_zero.make()


def run_puzzles():
    correct = 0
    n = 0
    for i, puzzle in enumerate(PUZZLES):
        try:
            print(f"Puzzle {i+1}/{len(PUZZLES)}: {puzzle.fen}")
            result = sigma_zero.play(puzzle.fen, TIME)
            move = result.get("move", "<unknown>")
            if move == puzzle.move:
                print("Correct!\n")
                correct += 1
            else:
                print(f"Incorrect. Got {move}, expected {puzzle.move}\n")
            n += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
    print(f"Score: {correct}/{n} = {correct/n*100:.2f}%")


if __name__ == "__main__":
    run_puzzles()
