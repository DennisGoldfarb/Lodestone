import pytest

torch = pytest.importorskip("torch")

from lodestone.data import one_hot, PAD_IDX, TOKEN_TO_IDX, VOCAB
from lodestone.model import LodestoneModel


def test_model_padding_invariance():
    seq = [TOKEN_TO_IDX["A"], TOKEN_TO_IDX["C"], TOKEN_TO_IDX["D"]]
    padded_seq = seq + [PAD_IDX, PAD_IDX]

    x_unpadded = one_hot(seq).unsqueeze(0)
    x_padded = one_hot(padded_seq).unsqueeze(0)

    model = LodestoneModel(d_model=16, nhead=4, num_layers=1, run_dim=8, num_runs=1)

    run_ids = torch.zeros(1, dtype=torch.long)

    logits_bias_unpadded, _ = model(x_unpadded, run_ids, return_bias=True)
    logits_bias_padded, _ = model(x_padded, run_ids, return_bias=True)

    torch.testing.assert_close(logits_bias_unpadded, logits_bias_padded)


def test_vocab_size_matches_embedding():
    model = LodestoneModel(d_model=8, nhead=1, num_layers=1, run_dim=4, num_runs=1)
    assert model.embed.in_features == len(VOCAB)
