import pandas as pd
import numpy as np

from lodestone.data import PeptideDataModule, PeptideDataset


def test_mz_mask(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "sequence": ["A", "AAAA"],
            "1": [1.0, 0.0],
            "2": [0.0, 1.0],
            "3": [0.0, 0.0],
            "4": [0.0, 0.0],
            "5": [0.0, 0.0],
            "dataset": ["run1", "run1"],
        }
    )
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    dm = PeptideDataModule(str(csv_path), batch_size=2)
    dm.setup("fit")
    out = capsys.readouterr().out
    assert "run1" in out
    assert "2 precursors" in out
    assert "0.0% charges masked" in out

    dataset = PeptideDataset(pd.read_csv(csv_path), dm.run_mapping)
    batch = [dataset[0], dataset[1]]
    x, y, run_ids, mask, seqs = dm.collate_fn(batch)

    expected_mask = np.ones((2, 5), dtype=float)
    assert np.array_equal(np.asarray(mask), expected_mask)
    assert seqs == ["A", "AAAA"]
