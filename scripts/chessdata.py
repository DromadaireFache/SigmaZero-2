import os
from datasets import load_dataset
from dataclasses import dataclass
from enum import Enum
import chess
from typing import Generator


class PosType(Enum):
    ALL = "all"
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


@dataclass
class Config:
    n: int = None
    postype: PosType = PosType.ALL
    balanced: bool = False
    eval_in_centipawns: bool = True
    repeat: bool = False


def get_dataset(config: Config = Config()):
    ds = load_dataset(
        "Lichess/chess-position-evaluations", split="train", streaming=True, token=os.getenv("HF_TOKEN", None)
    )
    hashset = set()

    for item in ds:
        h = hash(item["fen"])
        if h in hashset:
            continue
        hashset.add(h)
        item["fen"] = f"{item['fen'].strip()} 0 1"  # normalize fen
        if not fen_is_postype(item["fen"], config.postype):
            continue
        if config.balanced and item["cp"] is not None and abs(item["cp"]) > 50:
            continue
        if not config.eval_in_centipawns and item["cp"] is not None:
            item["cp"] = item["cp"] / 100.0  # convert to pawns
        yield item
        if config.repeat:
            yield item  # yield the same item again for repetition


def fens(config: Config = Config()) -> Generator[str, None, None]:
    count = 0
    for item in get_dataset(config):
        if item["fen"] is not None:
            count += 1
            yield item["fen"]
            if config.n is not None and count >= config.n:
                break


def fen_moves(config: Config = Config()) -> Generator[tuple[str, str], None, None]:
    count = 0
    for item in get_dataset(config):
        if item["fen"] is not None and item["line"] is not None:
            move = item["line"].split()[0] if item["line"] else None
            if move is not None:
                count += 1
                yield item["fen"], move
                if config.n is not None and count >= config.n:
                    break


def fen_evals(config: Config = Config()) -> Generator[tuple[str, int | float], None, None]:
    count = 0
    for item in get_dataset(config):
        if item["fen"] is not None and item["cp"] is not None:
            count += 1
            yield item["fen"], item["cp"]
            if config.n is not None and count >= config.n:
                break


def fen_mates(config: Config = Config()) -> Generator[tuple[str, int], None, None]:
    if config.balanced:
        raise ValueError("Mate positions cannot be balanced.")
    count = 0
    for item in get_dataset(config):
        if item["fen"] is not None and item["mate"] is not None:
            count += 1
            yield item["fen"], item["mate"]
            if config.n is not None and count >= config.n:
                break


def fen_is_postype(fen: str, postype: PosType) -> bool:
    return postype == PosType.ALL or get_postype(fen) == postype


starting_board = chess.Board()


def get_postype(fen: str) -> PosType:
    board = chess.Board(fen)
    if distance_from_starting_position(board) < 20:
        return PosType.OPENING
    weights = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    npm = sum(weights.get(piece.piece_type, 0) for piece in board.piece_map().values())
    if npm > 10:
        return PosType.MIDDLEGAME
    else:
        return PosType.ENDGAME


def distance_from_starting_position(board: chess.Board) -> int:
    return sum(1 for sq in chess.SQUARES if board.piece_at(sq) != starting_board.piece_at(sq))
