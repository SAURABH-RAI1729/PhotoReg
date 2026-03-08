import torch
import torch.nn as nn

# ============================================================
# models/deepsets.py
#
# DeepSets architecture for calorimeter regression.
#
# Key idea: treat each calorimeter cell as a "point" in a set.
# The model is permutation-invariant by design -- it doesn't
# care which order the 225 cells arrive in.
#
# Two sub-networks:
#
#   phi (per-cell):
#     Applied to each cell independently.
#     Input: 3 features per cell [energy, norm_row, norm_col]
#     Output: latent representation per cell
#
#   rho (global):
#     Takes the sum of all cell representations.
#     Sum pooling is permutation-invariant.
#     Output: single predicted energy value
#
# References:
#   Zaheer et al., "Deep Sets", NeurIPS 2017
# ============================================================


class DeepSets(nn.Module):

    def __init__(self, phi_dims=None, rho_dims=None, dropout=0.2):
        """
        phi_dims : list of hidden widths for the per-cell network
        rho_dims : list of hidden widths for the aggregation network
        dropout  : dropout probability
        """

        super().__init__()

        if phi_dims is None:
            phi_dims = [64, 128, 256]
        if rho_dims is None:
            rho_dims = [192, 64]

        # --------------------------------------------------
        # phi: per-cell feature extractor
        # --------------------------------------------------

        phi_layers = []
        in_dim = 3   # [energy, row, col]

        for h in phi_dims:
            phi_layers.append(nn.Linear(in_dim, h))
            phi_layers.append(nn.ReLU())
            phi_layers.append(nn.Dropout(p=dropout))
            in_dim = h

        self.phi = nn.Sequential(*phi_layers)

        # --------------------------------------------------
        # rho: regression head after aggregation
        # --------------------------------------------------

        rho_layers = []

        for h in rho_dims:
            rho_layers.append(nn.Linear(in_dim, h))
            rho_layers.append(nn.ReLU())
            rho_layers.append(nn.Dropout(p=dropout))
            in_dim = h

        # Final output layer (no activation for regression)
        rho_layers.append(nn.Linear(in_dim, 1))

        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x):
        """
        x : (batch, n_cells, 3)
        """

        # Apply phi to each cell independently
        # x: (batch, n_cells, phi_out)
        x = self.phi(x)

        # Sum-pool across cells
        # x: (batch, phi_out)
        x = torch.sum(x, dim=1)

        # Regression head
        x = self.rho(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = DeepSets()
    print(m)
    print(f"Trainable parameters: {count_parameters(m):,}")

    dummy = torch.randn(8, 225, 3)
    out = m(dummy)
    print(f"Output shape: {out.shape}")   # should be (8, 1)