import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ============================================================
# data/dataset.py
#
# Handles everything data-related:
#   - Loading the .npz file
#   - Train / validation split
#   - Feature normalization (fit on train only -- no leakage)
#   - PyTorch Dataset and DataLoader wrappers
#   - Saving / loading the fitted scaler (needed at inference)
# ============================================================

GRID = 15


class CalorimeterDataset(Dataset):
    """
    Wraps numpy arrays as a PyTorch Dataset.

    mode can be "flat", "image", or "deepsets" depending on
    which model will consume the data.
    """

    def __init__(self, X, y, mode="flat"):
        """
        X : numpy array, shape (N, 225), already normalized
        y : numpy array, shape (N,)
        mode : "flat"     -> tensor shape (N, 225)
               "image"    -> tensor shape (N, 1, 15, 15)
               "deepsets" -> tensor shape (N, 225, 3)  [energy, row, col]
        """

        self.mode = mode

        if mode == "flat":
            self.X = torch.tensor(X, dtype=torch.float32)

        elif mode == "image":
            X_img = X.reshape(-1, 1, GRID, GRID)
            self.X = torch.tensor(X_img, dtype=torch.float32)

        elif mode == "deepsets":
            self.X = torch.tensor(
                _build_deepsets_features(X), dtype=torch.float32
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _build_deepsets_features(X):
    """
    Build per-cell feature tensors for DeepSets.

    Each cell gets three features: [energy, norm_row, norm_col]
    Coordinates are normalized to [0, 1].

    Returns array of shape (N, 225, 3).
    """

    N = X.shape[0]

    coords = []
    for r in range(GRID):
        for c in range(GRID):
            coords.append([r / GRID, c / GRID])
    coords = np.array(coords, dtype=np.float32)   # (225, 2)

    events = np.zeros((N, GRID * GRID, 3), dtype=np.float32)
    events[:, :, 0] = X                            # energy
    events[:, :, 1] = coords[:, 0]                 # row
    events[:, :, 2] = coords[:, 1]                 # col

    return events


def load_data(data_path, val_split=0.2, seed=42, mode="flat"):
    """
    Full data pipeline: load -> split -> normalize -> wrap in Dataset.

    Returns
    -------
    train_dataset, val_dataset, scaler
    """

    # --------------------------------------------------------
    # Load raw arrays
    # --------------------------------------------------------

    data = np.load(data_path)
    X = data["X"]    # (N, 225)
    y = data["y"]    # (N,)

    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    print(f"  Energy range: [{y.min():.2f}, {y.max():.2f}] GeV")

    # --------------------------------------------------------
    # Train / validation split
    # Stratification not needed for continuous target, but we
    # fix the seed for full reproducibility.
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=seed
    )

    print(f"\nSplit: {X_train.shape[0]} train / {X_val.shape[0]} val")

    # --------------------------------------------------------
    # Normalize: fit scaler on training data only
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # --------------------------------------------------------
    # Build Datasets
    # --------------------------------------------------------

    train_dataset = CalorimeterDataset(X_train, y_train, mode=mode)
    val_dataset   = CalorimeterDataset(X_val,   y_val,   mode=mode)

    return train_dataset, val_dataset, scaler


def get_dataloaders(train_dataset, val_dataset, batch_size=128, num_workers=0):
    """
    Wrap datasets in DataLoaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {path}")


def load_scaler(path):
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler
