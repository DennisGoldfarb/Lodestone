import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd

try:  # pragma: no cover - fallback when PyTorch is unavailable
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
except Exception:  # pragma: no cover
    import numpy as np
    from types import SimpleNamespace
    import random
    import math as _math

    class Tensor(np.ndarray):
        def __new__(cls, data, dtype=None):
            arr = np.array(data, dtype=dtype)
            return arr.view(cls)

        def size(self, dim=None):
            return self.shape[dim] if dim is not None else self.shape

        def float(self):
            return Tensor(self.astype(np.float32))

        def long(self):
            return Tensor(self.astype(np.int64))

    def tensor(data, dtype=None):
        return Tensor(data, dtype=dtype)

    def stack(tensors):
        return Tensor(np.stack([np.asarray(t) for t in tensors]))

    def eye(n):
        return Tensor(np.eye(n))

    class Dataset:  # minimal dataset interface
        def __len__(self):
            raise NotImplementedError

        def __getitem__(self, idx):
            raise NotImplementedError

    class Subset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    class DataLoader:  # simple DataLoader yielding one batch
        def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, num_workers=0):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.collate_fn = collate_fn or (lambda x: x)

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = [self.dataset[j] for j in indices[i : i + self.batch_size]]
                yield self.collate_fn(batch)

    def random_split(dataset, lengths, generator=None):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subsets = []
        start = 0
        for length in lengths:
            subsets.append(Subset(dataset, indices[start : start + length]))
            start += length
        return subsets

    class Generator:
        def manual_seed(self, seed):
            random.seed(seed)
            return self

    torch = SimpleNamespace(
        tensor=tensor,
        stack=stack,
        eye=eye,
        Tensor=Tensor,
        float32=np.float32,
        long=np.int64,
        Generator=Generator,
        utils=SimpleNamespace(data=SimpleNamespace(Dataset=Dataset, DataLoader=DataLoader, random_split=random_split)),
    )

    Dataset = torch.utils.data.Dataset
    DataLoader = torch.utils.data.DataLoader
    random_split = torch.utils.data.random_split

try:  # pragma: no cover
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    class _LightningDataModule:
        pass

    pl = SimpleNamespace(LightningDataModule=_LightningDataModule)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["camC", "oxM", "ac-", "nemC"]
VOCAB = AMINO_ACIDS + SPECIAL_TOKENS
TOKEN_TO_IDX: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}

# Monoisotopic masses of amino acids and special tokens (in Daltons)
AA_MASS: Dict[str, float] = {
    "A": 71.03711,
    "C": 103.00919,
    "D": 115.02694,
    "E": 129.04259,
    "F": 147.06841,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "K": 128.09496,
    "L": 113.08406,
    "M": 131.04049,
    "N": 114.04293,
    "P": 97.05276,
    "Q": 128.05858,
    "R": 156.10111,
    "S": 87.03203,
    "T": 101.04768,
    "V": 99.06841,
    "W": 186.07931,
    "Y": 163.06333,
}

PROTON_MASS = 1.007276466812
WATER_MASS = 18.01056

SPECIAL_MASS: Dict[str, float] = {
    "camC": AA_MASS["C"] + 57.021464,
    "oxM": AA_MASS["M"] + 15.994915,
    "ac-": 42.010565,
    "nemC": AA_MASS["C"] + 125.047679,
}


def peptide_mass(seq: str) -> float:
    """Compute the monoisotopic mass of a peptide sequence."""
    mass = WATER_MASS
    i = 0
    while i < len(seq):
        if seq.startswith("camC", i):
            mass += SPECIAL_MASS["camC"]
            i += 4
        elif seq.startswith("oxM", i):
            mass += SPECIAL_MASS["oxM"]
            i += 3
        elif seq.startswith("ac-", i):
            mass += SPECIAL_MASS["ac-"]
            i += 3
        elif seq.startswith("nemC", i):
            mass += SPECIAL_MASS["nemC"]
            i += 4
        else:
            aa = seq[i]
            if aa not in AA_MASS:
                raise KeyError(f"Unknown residue {aa} in sequence {seq}")
            mass += AA_MASS[aa]
            i += 1
    return mass


def peptide_mz(seq: str, charge: int) -> float:
    mass = peptide_mass(seq)
    return (mass + PROTON_MASS * charge) / charge


def tokenize_sequence(seq: str) -> List[int]:
    tokens: List[int] = []
    i = 0
    while i < len(seq):
        if seq.startswith("camC", i):
            tokens.append(TOKEN_TO_IDX["camC"])
            i += 4
        elif seq.startswith("oxM", i):
            tokens.append(TOKEN_TO_IDX["oxM"])
            i += 3
        elif seq.startswith("ac-", i):
            tokens.append(TOKEN_TO_IDX["ac-"])
            i += 3
        elif seq.startswith("nemC", i):
            tokens.append(TOKEN_TO_IDX["nemC"])
            i += 4
        else:
            aa = seq[i]
            if aa not in TOKEN_TO_IDX:
                raise KeyError(f"Unknown token {aa} in sequence {seq}")
            tokens.append(TOKEN_TO_IDX[aa])
            i += 1
    return tokens


def one_hot(indices: List[int]) -> torch.Tensor:
    eye = torch.eye(len(VOCAB))
    return eye[indices]


@dataclass
class PeptideRecord:
    sequence: str
    charges: torch.Tensor
    run_id: int


class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame, run_mapping: Dict[str, int]):
        self.df = df
        self.run_mapping = run_mapping

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> PeptideRecord:
        row = self.df.iloc[idx]
        seq = row.iloc[0]
        charges = torch.tensor(row.iloc[1:6].values.astype(float), dtype=torch.float32)
        run_id = self.run_mapping[row.iloc[6]]
        return PeptideRecord(sequence=seq, charges=charges, run_id=run_id)


class PeptideDataModule(pl.LightningDataModule):
    def __init__(self, csv_path: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.run_mapping: Dict[str, int] = {}
        self.run_mz_ranges: Dict[int, Tuple[float, float]] = {}
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None
        self.test_set: Dataset | None = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.csv_path)
        run_names = sorted(df["dataset"].unique())
        self.run_mapping = {name: i for i, name in enumerate(run_names)}

        # Determine m/z range for each dataset based on non-zero charge states
        ranges: Dict[str, Tuple[float, float]] = {}
        for dataset_name, group in df.groupby("dataset"):
            mz_vals: List[float] = []
            for _, row in group.iterrows():
                seq = row.iloc[0]
                for charge_idx, val in enumerate(row.iloc[1:6].values, start=1):
                    if val > 0:
                        mz_vals.append(peptide_mz(seq, charge_idx))
            if mz_vals:
                ranges[dataset_name] = (min(mz_vals), max(mz_vals))
            else:
                ranges[dataset_name] = (0.0, float("inf"))

        self.run_mz_ranges = {
            self.run_mapping[name]: rng for name, rng in ranges.items()
        }

        # Print dataset statistics; masking disabled so report 0% masked
        for dataset_name, group in df.groupby("dataset"):
            total_precursors = len(group)
            print(
                f"{dataset_name}: {total_precursors} precursors, 0.0% charges masked"
            )

        dataset = PeptideDataset(df, self.run_mapping)
        n = len(dataset)
        n_train = int(0.7 * n)
        n_val = int(0.2 * n)
        n_test = n - n_train - n_val
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )

    def collate_fn(
        self, batch: List[PeptideRecord]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        seqs = [tokenize_sequence(rec.sequence) for rec in batch]
        max_len = max(len(s) for s in seqs)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        one_hot_seqs = torch.stack([one_hot(s) for s in padded])
        charges = torch.stack([rec.charges for rec in batch])
        run_ids_list = [rec.run_id for rec in batch]
        run_ids = torch.tensor(run_ids_list, dtype=torch.long)
        seq_strings = [rec.sequence for rec in batch]

        # Masking temporarily disabled: treat all charge states as valid
        mask = torch.ones_like(charges, dtype=torch.float32)

        return one_hot_seqs, charges, run_ids, mask, seq_strings

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
