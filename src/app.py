from dataclasses import dataclass
import os
import sys
import webview
import chess
import sigma_zero
import nnue


# Collect old bot versions from src/versions
VERSIONS_DIR = os.path.join(os.path.dirname(__file__), "versions")
bot_versions = [f[:-2] for f in os.listdir(VERSIONS_DIR)[::-1] if f.endswith(".c") and "const" not in f]


board = chess.Board()
white_captures = []
black_captures = []
bot_plays_white = False
bot_plays_black = False


@dataclass
class PreviousState:
    fen: str
    white_captures: list[str]
    black_captures: list[str]

    @staticmethod
    def current():
        return PreviousState(board.fen(), white_captures.copy(), black_captures.copy())


move_index = 0
previous_states = [PreviousState.current()]
evaluator = nnue.load_model("nnue_evaluator.pth")


def capture_diff() -> int:
    piece_values = {
        "P": 1,
        "N": 3,
        "B": 3,
        "R": 5,
        "Q": 9,
    }
    white_value = sum(piece_values.get(piece.upper(), 0) for piece in white_captures)
    black_value = sum(piece_values.get(piece.upper(), 0) for piece in black_captures)
    return white_value - black_value


class Api:
    def get_game_state(self):
        position: list[list[str | None]] = [[None for _ in range(8)] for _ in range(8)]

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                position[7 - chess.square_rank(square)][chess.square_file(square)] = piece.symbol()

        last_move = board.peek() if board.move_stack else None
        if last_move:
            last_move = {
                "from": {
                    "row": 7 - chess.square_rank(last_move.from_square),
                    "col": chess.square_file(last_move.from_square),
                },
                "to": {
                    "row": 7 - chess.square_rank(last_move.to_square),
                    "col": chess.square_file(last_move.to_square),
                },
            }

        return {
            "fen": board.fen(),
            "turn": "White" if board.turn else "Black",
            "position": position,
            "whiteCaptures": white_captures,
            "blackCaptures": black_captures,
            "captureDiff": capture_diff(),
            "gameOver": board.is_game_over(claim_draw=True),
            "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
            "lastMove": last_move,
        }

    def set_fen(self, fen: str) -> bool:
        global board, white_captures, black_captures, previous_states, move_index
        try:
            old_fen = board.fen()
            board.set_fen(fen)
            if board.is_valid():
                black_captures.clear()
                white_captures.clear()
                previous_states.clear()
                previous_states.append(PreviousState.current())
                move_index = 0
                return True
            else:
                board.set_fen(old_fen)
            return False
        except ValueError:
            return False

    def get_legal_moves(self, row: int, col: int) -> list[dict[str, int]]:
        square = chess.square(col, 7 - row)
        piece = board.piece_at(square)
        if piece is None or piece.color != board.turn:
            return []

        legal_moves = []
        for move in board.legal_moves:
            if move.from_square == square:
                to_row = 7 - chess.square_rank(move.to_square)
                to_col = chess.square_file(move.to_square)
                legal_moves.append({"row": to_row, "col": to_col})
        return legal_moves

    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int, promotion: str = None) -> bool:
        global board, white_captures, black_captures, previous_states, move_index
        from_square = chess.square(from_col, 7 - from_row)
        to_square = chess.square(to_col, 7 - to_row)
        move = chess.Move(from_square, to_square)
        
        # auto-promote to queen for simplicity
        if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
            if (board.piece_at(from_square).color == chess.WHITE and to_row == 0) or (
                board.piece_at(from_square).color == chess.BLACK and to_row == 7
            ):
                if promotion in ["q", "r", "b", "n"]:
                    promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
                    move.promotion = promo_map[promotion]
                else:
                    move.promotion = chess.QUEEN

        if move in board.legal_moves:
            captured_piece = board.piece_at(to_square)
            if captured_piece:
                if captured_piece.color == chess.WHITE:
                    white_captures.append(captured_piece.symbol())
                else:
                    black_captures.append(captured_piece.symbol())
            board.push(move)
            previous_states = previous_states[: move_index + 1] + [PreviousState.current()]
            move_index += 1
            print("NNUE Eval:", evaluator(nnue.fen_to_tensor(board.fen()).unsqueeze(0)).item())
            return True
        return False

    def reset_game(self) -> bool:
        global board, white_captures, black_captures, previous_states, move_index
        board = chess.Board()
        white_captures.clear()
        black_captures.clear()
        previous_states.clear()
        previous_states.append(PreviousState.current())
        move_index = 0
        return True

    def go_back(self) -> bool:
        global board, move_index, white_captures, black_captures, previous_states
        if move_index > 0:
            move_index -= 1
            board.set_fen(previous_states[move_index].fen)
            white_captures.clear()
            white_captures.extend(previous_states[move_index].white_captures)
            black_captures.clear()
            black_captures.extend(previous_states[move_index].black_captures)
            return True
        return False

    def go_forth(self) -> bool:
        global board, move_index, white_captures, black_captures, previous_states
        if move_index < len(previous_states) - 1:
            move_index += 1
            board.set_fen(previous_states[move_index].fen)
            white_captures.clear()
            white_captures.extend(previous_states[move_index].white_captures)
            black_captures.clear()
            black_captures.extend(previous_states[move_index].black_captures)
            return True
        return False

    def bot_move(self, millis: int, version: str, tries: int = 0) -> bool:
        if version.lower() == "latest":
            result = sigma_zero.play(board, millis)
        elif version.lower() == "aggressive":
            result = sigma_zero.fancy(board, millis)
        else:
            sigma_zero.make(version)
            result = sigma_zero.old_play(board, millis)
            
        move_uci = result.get("move", "<unknown>")
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                self.make_move(
                    7 - chess.square_rank(move.from_square),
                    chess.square_file(move.from_square),
                    7 - chess.square_rank(move.to_square),
                    chess.square_file(move.to_square),
                    promotion=move.promotion and chess.piece_symbol(move.promotion).lower(),
                )
            else:
                print("Bot move is not legal")
                sys.exit(1)
        except Exception:
            print(f"Failed to parse bot move: {result}")
            if tries < 3:
                return self.bot_move(millis, version, tries + 1)
            sys.exit(1)
        
        eval = result.get("eval", 0)
        if eval > 9000:
            eval = "Mate for white"
        elif eval < -9000:
            eval = "Mate for black"
        
        return {
            "depth": result.get("depth", 0),
            "eval": eval,
            "move": result.get("move", 0),
        }
    
    def get_bot_versions(self) -> list[str]:
        return bot_versions


api = Api()

if __name__ == "__main__":
    window = webview.create_window(
        "SigmaZero V2 Chess Engine", "index.html", js_api=api, width=900, height=690, resizable=False
    )
    webview.start()
