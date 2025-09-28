try:
    import torch
except Exception:  # pragma: no cover - fallback when torch unavailable
    from lodestone.data import torch

from lodestone.model import LodestoneModel


def is_unimodal(probs: torch.Tensor) -> bool:
    diffs = probs[1:] - probs[:-1]
    peak = torch.argmax(probs).item()
    left_ok = (diffs[:peak] >= -1e-7).all().item()
    right_ok = (diffs[peak:] <= 1e-7).all().item()
    return bool(left_ok and right_ok)


def test_split_normal_unimodal() -> None:
    model = LodestoneModel(d_model=4, nhead=1, num_layers=1, run_dim=2, num_runs=1, num_charge=5)
    mu = torch.tensor([2.0])
    sigma_l = torch.tensor([1.0])
    sigma_r = torch.tensor([1.0])
    logits = model._split_normal_logits(mu, sigma_l, sigma_r)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    assert is_unimodal(probs)
