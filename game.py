import chess
import sys
import src.sigma_zero as sigma_zero


# Unicode character mapping for chess pieces
PIECE_TO_UNICODE = {
    "r": "♖",
    "n": "♘",
    "b": "♗",
    "q": "♕",
    "k": "♔",
    "p": "♙",
    "R": "♜",
    "N": "♞",
    "B": "♝",
    "Q": "♛",
    "K": "♚",
    "P": "♟",
}


class Bot:
    def __init__(self):
        sigma_zero.make()
        try:
            self.time: int = int(input("Enter search time (ms, default=1000): "))
        except ValueError:
            self.time = 1000
        self.plays_white: bool = input("Play as white? (y/n, default=n): ").strip().lower() == "y"
        self.plays_black: bool = input("Play as black? (y/n, default=n): ").strip().lower() == "y"

    def get_move(self, board: chess.Board):
        if (board.turn == chess.WHITE and self.plays_white) or (
            board.turn == chess.BLACK and self.plays_black
        ):
            fen = board.fen()
            result = sigma_zero.play(fen, self.time)
            move_uci = result.get("move", "")

            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    print(f"Bot plays: {move_uci}")
                    return move
                else:
                    print("Bot suggested an illegal move. Please enter your move.")
                    print(result)
                    sys.exit(1)
            except Exception:
                print("Bot suggested an invalid move. Please enter your move.")
                print(result)
                sys.exit(1)

        while True:
            move_str = input(f"\n{('White' if board.turn else 'Black')} move: ").strip()
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
                else:
                    print("Illegal move. Try again.")
            except Exception:
                print("Invalid input. Use UCI format, e.g. e2e4.")


def print_board(board: chess.Board):
    import os

    os.system("cls" if os.name == "nt" else "clear")

    # ANSI color codes
    RESET = "\033[0m"
    WHITE_BG = "\033[48;5;15m"
    # BLACK_BG = "\033[48;5;0m"
    # WHITE_PIECE = "\033[38;5;15m"
    # BLACK_PIECE = "\033[38;5;236m"

    print("   a b c d e f g h")
    print("  +----------------+")
    board_str = str(board)
    rows = board_str.split("\n")
    for i, row in enumerate(rows):
        line = f"{8-i} |"
        for j, c in enumerate(row):
            if c == ".":
                bg = WHITE_BG if (i + j // 2) % 2 == 0 else ""
                line += f"{bg}  {RESET}"
            elif c.isalpha():
                line += f"{PIECE_TO_UNICODE.get(c,' ')} "
        line += f"| {8-i}"
        print(line)
    print("  +----------------+")
    print("   a b c d e f g h")


def main(fen: str = chess.STARTING_FEN):
    bot = Bot()
    board = chess.Board(fen)
    while not board.is_game_over():
        print_board(board)
        move = bot.get_move(board)
        board.push(move)
    print_board(board)
    result = board.result()
    print(f"Game over! Result: {result}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
