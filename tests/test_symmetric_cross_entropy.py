import torch
from lodestone.losses import symmetric_cross_entropy


def test_symmetric_cross_entropy_matches_manual():
    preds = torch.log(torch.tensor([0.7, 0.3]))
    target = torch.tensor([0.6, 0.4])
    loss = symmetric_cross_entropy(preds, target)

    manual = -(
        target * torch.log(torch.tensor([0.7, 0.3]))
        + torch.tensor([0.7, 0.3]) * torch.log(target)
    )
    assert torch.allclose(loss, manual)


def test_symmetric_cross_entropy_is_symmetric():
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.6, 0.4])
    loss_pq = symmetric_cross_entropy(torch.log(p), q).sum()
    loss_qp = symmetric_cross_entropy(torch.log(q), p).sum()
    assert torch.allclose(loss_pq, loss_qp)
