import torch
import torch.nn as nn

# ============================================================
# models/cnn.py
# I would skip the reference here ;)
# Treats the 15x15 calorimeter as a single-channel 2D image.
# Convolutional layers capture local spatial correlations
# in the shower shape before the regression head.
#
# Architecture:
#   Conv2d(1, 16, 3, pad=1) → ReLU → MaxPool  [15→7]
#   Conv2d(16, 32, 3, pad=1)→ ReLU → MaxPool  [7→3]
#   Conv2d(32, 64, 3, pad=1)→ ReLU
#   Flatten → Linear(576, 128) → ReLU → Dropout
#           → Linear(128, 64)  → ReLU → Dropout
#           → Linear(64, 1)
# ============================================================


class CalorimeterCNN(nn.Module):

    def __init__(self, channels=None, dropout=0.2):
        """
        channels : list of [c1, c2, c3] output channels for the 3 conv layers
        dropout  : dropout probability in FC layers
        """

        super().__init__()

        if channels is None:
            channels = [16, 32, 64]

        c1, c2, c3 = channels

        # --------------------------------------------------
        # Convolutional feature extractor
        # --------------------------------------------------

        self.conv = nn.Sequential(

            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 15 -> 7

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 7  -> 3

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU()
            # No further pooling: spatial size stays at 3x3
        )

        # After the 3 conv layers: feature map is (c3, 3, 3)
        flat_dim = c3 * 3 * 3

        # --------------------------------------------------
        # Fully connected regression head
        # --------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x : (batch, 1, 15, 15)
        """

        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = CalorimeterCNN()
    print(m)
    print(f"Trainable parameters: {count_parameters(m):,}")

    dummy = torch.randn(8, 1, 15, 15)
    out = m(dummy)
    print(f"Output shape: {out.shape}")   # should be (8, 1)
