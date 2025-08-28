from pathlib import Path
import pandas as pd

from lodestone.concat_csd import collect_csd


def create_dummy_csv(dir_path: Path, seq: str, val: int) -> None:
    df = pd.DataFrame({"sequence": [seq], "value": [val]})
    df.to_csv(dir_path / "csd.csv", index=False)


def test_collect_csd(tmp_path: Path) -> None:
    d1 = tmp_path / "FolderX" / "dataset1"
    d2 = tmp_path / "FolderY" / "dataset2"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)

    create_dummy_csv(d1, "AAA", 1)
    create_dummy_csv(d2, "BBB", 2)

    combined = collect_csd(tmp_path)

    assert set(combined["dataset"]) == {"dataset1", "dataset2"}
    assert len(combined) == 2
    assert (
        combined.loc[combined["sequence"] == "AAA", "dataset"].item() == "dataset1"
    )
    assert (
        combined.loc[combined["sequence"] == "BBB", "dataset"].item() == "dataset2"
    )
