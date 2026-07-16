import torch
from torch import nn


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted CrossEntropyLoss for ASVspoof binary classification.

    Labels:
        0 -> spoof
        1 -> bonafide
    """

    def __init__(self, spoof_weight=1.0, bonafide_weight=8.0):
        super().__init__()
        self.spoof_weight = spoof_weight
        self.bonafide_weight = bonafide_weight

    def forward(self, logits, labels, **batch):
        weights = torch.tensor(
            [self.spoof_weight, self.bonafide_weight],
            dtype=logits.dtype,
            device=logits.device,
        )

        loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=weights,
        )

        return {"loss": loss}
