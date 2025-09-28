try:
    import torch
except Exception:  # pragma: no cover - fallback when torch unavailable
    from lodestone.data import torch

from lodestone.data import one_hot, PAD_IDX, TOKEN_TO_IDX
from lodestone.model import LodestoneModel


def _model_for_test() -> LodestoneModel:
    # Small model for test purposes; use deterministic evaluation mode.
    model = LodestoneModel(
        d_model=16,
        nhead=4,
        num_layers=1,
        run_dim=8,
        num_runs=1,
    )
    model.eval()
    return model


def test_padded_and_unpadded_bias_logits_match():
    model = _model_for_test()
    run_ids = torch.tensor([0])

    seq_tokens = [TOKEN_TO_IDX["A"], TOKEN_TO_IDX["C"]]
    unpadded = one_hot(seq_tokens).unsqueeze(0)

    padded_tokens = seq_tokens + [PAD_IDX, PAD_IDX]
    padded = one_hot(padded_tokens).unsqueeze(0)

    logits_unpadded, _ = model(unpadded, run_ids, return_bias=True)
    logits_padded, _ = model(padded, run_ids, return_bias=True)

    assert torch.allclose(logits_unpadded, logits_padded, atol=1e-6)
