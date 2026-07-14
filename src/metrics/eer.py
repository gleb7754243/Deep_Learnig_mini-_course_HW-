import torch

from src.metrics.base_metric import BaseMetric


def compute_eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Equal Error Rate.

    labels:
        1 -> bonafide
        0 -> spoof

    scores:
        higher score means more likely bonafide
    """
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()

    bonafide_mask = labels == 1
    spoof_mask = labels == 0

    n_bonafide = bonafide_mask.sum().item()
    n_spoof = spoof_mask.sum().item()

    if n_bonafide == 0 or n_spoof == 0:
        return 0.0

    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    accepted_bonafide = torch.cumsum((sorted_labels == 1).float(), dim=0)
    accepted_spoof = torch.cumsum((sorted_labels == 0).float(), dim=0)

    false_reject_rate = (n_bonafide - accepted_bonafide) / n_bonafide
    false_accept_rate = accepted_spoof / n_spoof

    # Add the strict-threshold point: accept nobody.
    false_reject_rate = torch.cat(
        [torch.ones(1), false_reject_rate]
    )
    false_accept_rate = torch.cat(
        [torch.zeros(1), false_accept_rate]
    )

    diff = torch.abs(false_accept_rate - false_reject_rate)
    best_index = torch.argmin(diff)

    eer = (
        false_accept_rate[best_index]
        + false_reject_rate[best_index]
    ) / 2.0

    return float(eer.item())


class EERMetric(BaseMetric):
    """
    Batch-level EER metric for logging during training.

    For final reporting, full-set EER must be computed from all eval scores.
    """

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        probabilities = torch.softmax(logits, dim=-1)
        bonafide_scores = probabilities[:, 1]
        return compute_eer(bonafide_scores, labels)
