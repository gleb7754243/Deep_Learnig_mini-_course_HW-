import torch
from torch import nn


class MFMConv2d(nn.Module):
    """
    Max-Feature-Map convolution layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv(x)
        first_half, second_half = torch.chunk(x, chunks=2, dim=1)
        return torch.maximum(first_half, second_half)


class MFMLinear(nn.Module):
    """
    Max-Feature-Map linear layer.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        x = self.linear(x)
        first_half, second_half = torch.chunk(x, chunks=2, dim=1)
        return torch.maximum(first_half, second_half)


class LightCNN(nn.Module):
    """
    LightCNN / LCNN-style model for ASVspoof spectrogram classification.

    Expected input:
        [batch_size, 1, 257, 600]

    Output:
        logits with shape [batch_size, n_class]
    """

    def __init__(
        self,
        in_channels=1,
        n_class=2,
        base_channels=16,
        embedding_dim=128,
        dropout_p=0.2,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.features = nn.Sequential(
            MFMConv2d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            MFMConv2d(base_channels, base_channels * 2, kernel_size=1),
            nn.BatchNorm2d(base_channels * 2),
            MFMConv2d(base_channels * 2, base_channels * 3, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 3),

            MFMConv2d(base_channels * 3, base_channels * 4, kernel_size=1),
            nn.BatchNorm2d(base_channels * 4),
            MFMConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            MFMConv2d(base_channels * 4, base_channels * 4, kernel_size=1),
            nn.BatchNorm2d(base_channels * 4),
            MFMConv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = MFMLinear(base_channels * 2, embedding_dim)

        # Required by homework: dropout before final BatchNorm.
        self.dropout = nn.Dropout(p=dropout_p)
        self.final_bn = nn.BatchNorm1d(embedding_dim)

        self.classifier = nn.Linear(embedding_dim, n_class)

    def _prepare_input(self, data_object):
        """
        Convert input to [batch, channels, freq, time].

        Some template paths may pass [channels, batch, freq, time],
        so we handle this safely here.
        """
        if data_object.dim() == 3:
            data_object = data_object.unsqueeze(1)

        if data_object.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(data_object.shape)}")

        if data_object.shape[1] != self.in_channels and data_object.shape[0] == self.in_channels:
            data_object = data_object.permute(1, 0, 2, 3).contiguous()

        return data_object.float()

    def forward(self, data_object, **batch):
        x = self._prepare_input(data_object)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.final_bn(x)
        logits = self.classifier(x)

        return {"logits": logits}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"

        return result_info
