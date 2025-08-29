import torch
import torch.nn.functional as F


def symmetric_cross_entropy(
    preds: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Compute the symmetric cross-entropy between ``preds`` and ``target``.

    ``preds`` are raw, unnormalized logits while ``target`` is a probability
    distribution over the same classes.  The loss is a weighted sum of the
    forward cross entropy (``CE(target, preds)``) and the reverse cross entropy
    (``CE(preds, target)``).

    Parameters
    ----------
    preds:
        Unnormalized model predictions of shape ``[*, C]``.
    target:
        Ground-truth probability distribution of the same shape.
    alpha:
        Weight for the standard cross entropy term.
    beta:
        Weight for the reverse cross entropy term.
    epsilon:
        Small constant for numerical stability when taking ``log`` of
        ``target``.

    Returns
    -------
    torch.Tensor
        Element-wise symmetric cross-entropy loss with the same shape as
        ``target``.
    """

    log_probs = F.log_softmax(preds, dim=-1)
    probs = log_probs.exp()

    ce = -(target * log_probs)
    rce = -(probs * torch.log(target.clamp_min(epsilon)))
    return alpha * ce + beta * rce
