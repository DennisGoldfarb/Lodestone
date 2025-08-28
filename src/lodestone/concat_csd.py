from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd



def collect_csd(root: str | Path) -> pd.DataFrame:
    """Collect all ``csd.csv`` files under ``root``.

    The dataset name is inferred from the parent directory name of each
    ``csd.csv`` file and stored in a new ``dataset`` column.

    Parameters
    ----------
    root:
        Root directory to search.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe containing data from all discovered files.
    """
    root_path = Path(root)
    frames: List[pd.DataFrame] = []
    for csv_path in root_path.rglob("csd.csv"):
        dataset_name = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df["dataset"] = dataset_name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No csd.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate csd.csv files and add dataset column")
    parser.add_argument("root", help="Root directory to search for csd.csv files")
    parser.add_argument("output", help="Where to write the concatenated CSV")
    args = parser.parse_args()

    df = collect_csd(args.root)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
