import torch
import torch.nn as nn

# ============================================================
# models/mlp.py
#
# A small fully-connected network for photon energy regression.
#
# Dropout is included deliberately -- it serves double duty:
#   1. Regularization during training
#   2. Monte Carlo Dropout uncertainty estimation at inference
#      (keep dropout ON during inference, run N forward passes)
#
# Architecture:
#   Input (225)
#     → Linear(225, 256) → BatchNorm → ReLU → Dropout
#     → Linear(256, 128) → BatchNorm → ReLU → Dropout
#     → Linear(128, 64)  → BatchNorm → ReLU → Dropout
#     → Linear(64, 1)
# ============================================================


class PhotonEnergyMLP(nn.Module):

    def __init__(self, hidden_dims=None, dropout=0.2):
        """
        hidden_dims : list of hidden layer widths
        dropout     : dropout probability (same across all layers)
        """

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        in_dim = 225

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = PhotonEnergyMLP()
    print(m)
    print(f"Trainable parameters: {count_parameters(m):,}")

    # Quick shape check
    dummy = torch.randn(8, 225)
    out = m(dummy)
    print(f"Output shape: {out.shape}")   # should be (8, 1)
