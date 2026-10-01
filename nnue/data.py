import glob
import math
import os
import shutil
import time
from typing import Optional
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from numba import njit

from .archs.chessnn import ChessNN


HF_DATASET_NAME = "Lichess/chess-position-evaluations"
DEFAULT_HF_DATASET_DIR = "data/hf_chess_position_evaluations"
ENCODED_CACHE_DIR = "encoded"

def _local_arrow_files(dataset_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dataset_dir, "**/*.arrow"), recursive=True))


def _count_arrow_rows(file_paths: list[str]) -> int:
    if not file_paths:
        return 0
    total_rows = 0
    for file_path in tqdm(file_paths, desc="Counting rows", unit="file"):
        with pa.memory_map(file_path, "r") as source:
            reader = ipc.open_stream(source)
            for record_batch in reader:
                total_rows += record_batch.num_rows
    return total_rows


def _iter_arrow_record_batches(file_path: str):
    with pa.memory_map(file_path, "r") as source:
        reader = ipc.open_stream(source)
        for record_batch in reader:
            yield record_batch


def _encoded_cache_paths(dataset_dir: str, file_index: int) -> tuple[str, str]:
    cache_dir = os.path.join(dataset_dir, ENCODED_CACHE_DIR)
    stem = f"{file_index:06d}"
    return (
        os.path.join(cache_dir, f"{stem}.inputs.npy"),
        os.path.join(cache_dir, f"{stem}.targets.npy"),
    )


def ensure_encoded_dataset(dataset_dir: str, chess_nn: ChessNN):
    """Pre-encode FENs once so training does not parse them every epoch."""
    cache_files = _local_arrow_files(dataset_dir)
    cache_dir = os.path.join(dataset_dir, ENCODED_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    for file_index, file_path in enumerate(tqdm(cache_files, desc="Encoding positions", unit="file")):
        inputs_path, targets_path = _encoded_cache_paths(dataset_dir, file_index)
        if os.path.exists(inputs_path) and os.path.exists(targets_path):
            continue

        valid_rows = 0
        for record_batch in _iter_arrow_record_batches(file_path):
            cp_values = record_batch.column(record_batch.schema.get_field_index("cp")).to_pylist()
            mate_values = record_batch.column(record_batch.schema.get_field_index("mate")).to_pylist()
            valid_rows += sum(parse_eval(cp, mate) is not None for cp, mate in zip(cp_values, mate_values))

        temp_inputs_path = f"{inputs_path}.tmp"
        temp_targets_path = f"{targets_path}.tmp"
        encoded_inputs = np.lib.format.open_memmap(
            temp_inputs_path, mode="w+", dtype=np.uint8, shape=(valid_rows, 769)
        )
        encoded_targets = np.lib.format.open_memmap(
            temp_targets_path, mode="w+", dtype=np.float32, shape=(valid_rows,)
        )

        row_index = 0
        for record_batch in _iter_arrow_record_batches(file_path):
            cp_index = record_batch.schema.get_field_index("cp")
            mate_index = record_batch.schema.get_field_index("mate")
            fen_index = record_batch.schema.get_field_index("fen")
            cp_values = record_batch.column(cp_index).to_pylist()
            mate_values = record_batch.column(mate_index).to_pylist()
            fen_values = record_batch.column(fen_index).to_pylist()
            for cp, mate, fen in zip(cp_values, mate_values, fen_values):
                parsed_eval = parse_eval(cp, mate)
                if parsed_eval is None:
                    continue
                encoded_inputs[row_index] = chess_nn.fen_to_input(fen).numpy()
                encoded_targets[row_index] = parsed_eval
                row_index += 1

        encoded_inputs.flush()
        encoded_targets.flush()
        del encoded_inputs, encoded_targets
        os.replace(temp_inputs_path, inputs_path)
        os.replace(temp_targets_path, targets_path)
            

@njit
def parse_eval(cp: Optional[int], mate: Optional[int]) -> Optional[float]:
    if mate is not None:
        # Convert mate-in-N to a large eval that decays with distance
        # mate-in-1 -> 2400cp, mate-in-10 -> 1770cp, mate-in-30 -> 1200cp, ...
        sign = 1 if mate > 0 else -1
        cp = sign * (10 + 15 * math.exp(-abs(mate) / 15)) * 100
    elif cp is None:
        return None
    return max(min(cp / 100.0, 25.0), -25.0)


class HFDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        chess_nn: ChessNN,
        split: str,
        max_samples=None,
        seed=0,
        val_fraction: float = 0.2,
        dataset_dir: str = DEFAULT_HF_DATASET_DIR,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.chess_nn = chess_nn
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.val_fraction = val_fraction
        self.dataset_dir = dataset_dir
        self.cache_files = _local_arrow_files(self.dataset_dir)

        if not self.cache_files:
            raise FileNotFoundError(f"No local Arrow files found in {self.dataset_dir}")

        self.split_every = max(int(round(1.0 / self.val_fraction)), 1)
        self.split_files = [
            file_path
            for file_index, file_path in enumerate(self.cache_files)
            if (file_index % self.split_every == 0) == (self.split == "val")
        ]
        self.split_file_indices = [self.cache_files.index(file_path) for file_path in self.split_files]

        if max_samples is None:
            self.total_rows = _count_arrow_rows(self.split_files)
        else:
            self.total_rows = max_samples

    def _split_limit(self):
        if self.max_samples is None:
            return None
        val_size = int(self.max_samples * self.val_fraction)
        train_size = self.max_samples - val_size
        return train_size if self.split == "train" else val_size

    def __len__(self):
        total = self.total_rows

        if self.max_samples is not None:
            split_limit = self._split_limit()
            return split_limit if split_limit is not None else total
        return total

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        worker_file_indices = self.split_file_indices[worker_id::num_workers]
        if not worker_file_indices:
            return

        limit = None
        split_limit = self._split_limit()
        if split_limit is not None:
            active_workers = min(num_workers, len(self.split_files))
            base = split_limit // active_workers
            rem = split_limit % active_workers
            limit = base + (1 if worker_id < rem else 0)

        produced = 0
        for file_index in worker_file_indices:
            inputs_path, targets_path = _encoded_cache_paths(self.dataset_dir, file_index)
            inputs_array = np.load(inputs_path, mmap_mode="r")
            targets_array = np.load(targets_path, mmap_mode="r")
            remaining = None if limit is None else limit - produced
            if remaining is not None and remaining <= 0:
                return
            end = len(inputs_array) if remaining is None else min(len(inputs_array), remaining)
            inputs = torch.from_numpy(inputs_array[:end].copy()).to(dtype=torch.float32)
            targets = torch.from_numpy(targets_array[:end].copy())
            if end:
                yield inputs, targets, np.float32(0.0), np.float32(0.0)
                produced += end
            if limit is not None and produced >= limit:
                return


def download_hf_dataset(dataset_dir: str, force: bool = False):
    if os.path.isdir(dataset_dir):
        if force:
            print(f"Removing existing local dataset at {dataset_dir} before re-download")
            shutil.rmtree(dataset_dir)
        else:
            arrow_files = _local_arrow_files(dataset_dir)
            if arrow_files:
                print(f"Local dataset already exists at {dataset_dir}")
                return
            print(f"Local dataset directory exists but contains no Arrow files: {dataset_dir}")
            shutil.rmtree(dataset_dir)

    print(f"Downloading {HF_DATASET_NAME} to {dataset_dir}...")
    dataset = load_dataset(HF_DATASET_NAME, split="train", streaming=False, token=os.getenv("HF_TOKEN", None))
    os.makedirs(os.path.dirname(dataset_dir) or ".", exist_ok=True)
    dataset.save_to_disk(dataset_dir)
    print(f"Saved local dataset to {dataset_dir}")


def ensure_local_dataset(dataset_dir: str):
    if not _local_arrow_files(dataset_dir):
        download_hf_dataset(dataset_dir)