import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["camC", "oxM", "ac-", "nemC"]
VOCAB = AMINO_ACIDS + SPECIAL_TOKENS
TOKEN_TO_IDX: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}


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
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None
        self.test_set: Dataset | None = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.csv_path)
        run_names = sorted(df["dataset"].unique())
        self.run_mapping = {name: i for i, name in enumerate(run_names)}

        dataset = PeptideDataset(df, self.run_mapping)
        n = len(dataset)
        n_train = int(0.7 * n)
        n_val = int(0.2 * n)
        n_test = n - n_train - n_val
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )

    def collate_fn(self, batch: List[PeptideRecord]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = [tokenize_sequence(rec.sequence) for rec in batch]
        max_len = max(len(s) for s in seqs)
        padded = [s + [0] * (max_len - len(s)) for s in seqs]
        one_hot_seqs = torch.stack([one_hot(s) for s in padded])
        charges = torch.stack([rec.charges for rec in batch])
        run_ids = torch.tensor([rec.run_id for rec in batch], dtype=torch.long)
        return one_hot_seqs, charges, run_ids

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
