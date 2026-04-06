import os
import csv
from datasets import load_dataset
from dataclasses import dataclass
from enum import Enum
import chess
from typing import Generator, Any


CACHE_FILE = "data/lichess-positions.csv"
CACHE_LIMIT = 1_000_000
CACHE_FIELDS = ["fen", "line", "cp", "mate"]


class PosType(Enum):
    ALL = "all"
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


@dataclass
class Dataloader:
    n: int = None
    postype: PosType = PosType.ALL
    balanced: bool = False
    eval_in_centipawns: bool = True
    repeat: bool = False

    def get_dataset(self):
        items = _load_cached_items()
        hashset = set()

        try:
            for item in items:
                normalized = _normalize_item(item, append_move_count=False)
                h = hash(normalized["fen"])
                if h in hashset:
                    continue
                hashset.add(h)
                if not chess.Board(normalized["fen"]).is_valid():
                    continue
                prepared = self._prepare_item_for_output(normalized)
                if prepared is None:
                    continue
                yield prepared
                if self.repeat:
                    yield prepared

            ds = load_dataset(
                "Lichess/chess-position-evaluations",
                split="train",
                streaming=True,
                token=os.getenv("HF_TOKEN", None),
            )

            for item in ds:
                if item.get("fen") is None:
                    continue

                normalized = _normalize_item(item, append_move_count=True)
                h = hash(normalized["fen"])
                if h in hashset:
                    continue
                hashset.add(h)
                if not chess.Board(normalized["fen"]).is_valid():
                    continue

                if len(items) < CACHE_LIMIT:
                    items.append(
                        {
                            "fen": normalized.get("fen"),
                            "line": normalized.get("line"),
                            "cp": normalized.get("cp"),
                            "mate": normalized.get("mate"),
                        }
                    )

                prepared = self._prepare_item_for_output(normalized)
                if prepared is None:
                    continue
                yield prepared
                if self.repeat:
                    yield prepared
        finally:
            _save_cached_items(items)

    def _prepare_item_for_output(self, item: dict[str, Any]) -> dict[str, Any] | None:
        if item.get("fen") is None:
            return None
        if not fen_is_postype(item["fen"], self.postype):
            return None
        if self.balanced and item.get("cp") is not None and abs(item["cp"]) > 50:
            return None

        prepared = {
            "fen": item.get("fen"),
            "line": item.get("line"),
            "cp": item.get("cp"),
            "mate": item.get("mate"),
        }
        if not self.eval_in_centipawns and prepared["cp"] is not None:
            prepared["cp"] = prepared["cp"] / 100.0
        return prepared

    def fens(self) -> Generator[str, None, None]:
        count = 0
        for item in self.get_dataset():
            if item["fen"] is not None:
                count += 1
                yield item["fen"]
                if self.n is not None and count >= self.n:
                    break

    def fen_moves(self) -> Generator[tuple[str, str], None, None]:
        count = 0
        for item in self.get_dataset():
            if item["fen"] is not None and item["line"] is not None:
                move = item["line"].split()[0] if item["line"] else None
                if move is not None:
                    count += 1
                    yield item["fen"], move
                    if self.n is not None and count >= self.n:
                        break

    def fen_evals(self) -> Generator[tuple[str, int | float], None, None]:
        count = 0
        for item in self.get_dataset():
            if item["fen"] is not None and item["cp"] is not None:
                count += 1
                yield item["fen"], item["cp"]
                if self.n is not None and count >= self.n:
                    break

    def fen_mates(self) -> Generator[tuple[str, int], None, None]:
        if self.balanced:
            raise ValueError("Mate positions cannot be balanced.")
        count = 0
        for item in self.get_dataset():
            if item["fen"] is not None and item["mate"] is not None:
                count += 1
                yield item["fen"], item["mate"]
                if self.n is not None and count >= self.n:
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


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    return int(value)


def _normalize_item(item: dict[str, Any], append_move_count: bool = False) -> dict[str, Any]:
    fen = item.get("fen")
    if fen is None:
        return {"fen": None, "line": item.get("line"), "cp": item.get("cp"), "mate": item.get("mate")}

    cp = item.get("cp")
    mate = item.get("mate")
    if isinstance(cp, str):
        cp = _parse_optional_int(cp)
    if isinstance(mate, str):
        mate = _parse_optional_int(mate)
    
    fen = fen.strip()
    if append_move_count:
        postype = get_postype(fen)
        if postype == PosType.MIDDLEGAME:
            fen += " 0 30"
        elif postype == PosType.ENDGAME:
            fen += " 0 60"
        else:
            fen += " 0 1"

    return {
        "fen": fen,
        "line": item.get("line"),
        "cp": cp,
        "mate": mate,
    }


def _load_cached_items() -> list[dict[str, Any]]:
    if not os.path.exists(CACHE_FILE):
        return []

    items = []
    with open(CACHE_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            items.append(
                {
                    "fen": row.get("fen"),
                    "line": row.get("line") or None,
                    "cp": _parse_optional_int(row.get("cp")),
                    "mate": _parse_optional_int(row.get("mate")),
                }
            )
    return items


def _save_cached_items(items: list[dict[str, Any]]) -> None:
    with open(CACHE_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "fen": item.get("fen") or "",
                    "line": item.get("line").split()[0] or "",
                    "cp": "" if item.get("cp") is None else item.get("cp"),
                    "mate": "" if item.get("mate") is None else item.get("mate"),
                }
            )
