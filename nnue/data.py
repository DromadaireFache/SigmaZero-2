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


MISSING_VALUE = np.iinfo(np.int64).min


@njit
def parse_eval_batch(cp_values: np.ndarray, mate_values: np.ndarray):
    targets = np.empty(cp_values.shape[0], dtype=np.float32)
    valid = np.ones(cp_values.shape[0], dtype=np.bool_)
    for index in range(cp_values.shape[0]):
        cp = cp_values[index]
        mate = mate_values[index]
        if mate != MISSING_VALUE:
            sign = 1 if mate > 0 else -1
            targets[index] = max(min(sign * (10 + 15 * math.exp(-abs(mate) / 15)) , 25.0), -25.0)
        elif cp != MISSING_VALUE:
            targets[index] = max(min(cp / 100.0, 25.0), -25.0)
        else:
            targets[index] = 0.0
            valid[index] = False
    return targets, valid


class HFDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        chess_nn: ChessNN,
        split: str,
        max_samples=None,
        epoch_samples=None,
        seed=0,
        val_fraction: float = 0.2,
        dataset_dir: str = DEFAULT_HF_DATASET_DIR,
    ):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.chess_nn = chess_nn
        self.split = split
        self.max_samples = max_samples
        self.epoch_samples = epoch_samples
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

        if max_samples is None:
            self.total_rows = _count_arrow_rows(self.split_files)
        else:
            self.total_rows = max_samples

    def _split_limit(self):
        if self.epoch_samples is not None:
            return self.epoch_samples
        if self.max_samples is None:
            return None
        val_size = int(self.max_samples * self.val_fraction)
        train_size = self.max_samples - val_size
        return train_size if self.split == "train" else val_size

    def __len__(self):
        total = self.total_rows

        if self.epoch_samples is not None:
            return self.epoch_samples
        if self.max_samples is not None:
            split_limit = self._split_limit()
            return split_limit if split_limit is not None else total
        return total

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        epoch = getattr(self, "_epoch", 0)
        self._epoch = epoch + 1
        file_order = list(self.split_files)
        rng = np.random.default_rng(self.seed + epoch)
        if self.epoch_samples is not None:
            rng.shuffle(file_order)
        worker_files = file_order[worker_id::num_workers]
        if not worker_files:
            return

        limit = None
        split_limit = self._split_limit()
        if split_limit is not None:
            active_workers = min(num_workers, len(self.split_files))
            base = split_limit // active_workers
            rem = split_limit % active_workers
            limit = base + (1 if worker_id < rem else 0)

        produced = 0
        pending_fetch = 0.0
        pending_norm = 0.0
        for file_path in worker_files:
            file_fetch_start = time.perf_counter()
            for record_batch in _iter_arrow_record_batches(file_path):
                fetch_time = time.perf_counter() - file_fetch_start
                pending_fetch += fetch_time

                parse_start = time.perf_counter()
                cp_index = record_batch.schema.get_field_index("cp")
                mate_index = record_batch.schema.get_field_index("mate")
                fen_index = record_batch.schema.get_field_index("fen")
                if cp_index < 0 or mate_index < 0 or fen_index < 0:
                    raise ValueError(f"Arrow batch schema missing required fields: {record_batch.schema}")
                cp_values = record_batch.column(cp_index).to_pylist()
                mate_values = record_batch.column(mate_index).to_pylist()
                fen_values = record_batch.column(fen_index).to_pylist()
                cp_array = np.asarray(
                    [MISSING_VALUE if value is None else value for value in cp_values], dtype=np.int64
                )
                mate_array = np.asarray(
                    [MISSING_VALUE if value is None else value for value in mate_values], dtype=np.int64
                )
                targets, valid = parse_eval_batch(cp_array, mate_array)
                valid_indices = np.flatnonzero(valid)
                if self.epoch_samples is not None:
                    rng.shuffle(valid_indices)
                if limit is not None:
                    remaining = limit - produced
                    if remaining <= 0:
                        return
                    valid_indices = valid_indices[:remaining]
                selected_fens = [fen_values[index] for index in valid_indices]

                if not selected_fens:
                    pending_norm += time.perf_counter() - parse_start
                    file_fetch_start = time.perf_counter()
                    continue

                inputs = torch.from_numpy(self.chess_nn.fens_to_input(selected_fens))
                targets = torch.from_numpy(targets[valid_indices].copy())
                pending_norm += time.perf_counter() - parse_start
                file_fetch_start = time.perf_counter()

                yield inputs, targets, np.float32(pending_fetch), np.float32(pending_norm)
                pending_fetch = 0.0
                pending_norm = 0.0

                produced += inputs.size(0)
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