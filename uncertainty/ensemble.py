import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.mlp      import PhotonEnergyMLP
from models.cnn      import CalorimeterCNN
from models.deepsets import DeepSets

# ============================================================
# uncertainty/ensemble.py
#
# Deep Ensemble uncertainty estimation.
#
# Train N_MODELS independent copies of the same architecture
# with different random seeds. At inference time, the spread
# of predictions across models estimates epistemic uncertainty.
#
# This is arguably more reliable than MC Dropout because each
# model explores a different region of weight space.
#
# Reference:
#   Lakshminarayanan et al., "Simple and Scalable Predictive
#   Uncertainty Estimation using Deep Ensembles", NeurIPS 2017
# ============================================================

N_EPOCHS = 60    # shorter than main training -- ensemble members
                  # don't need to be fully converged individually


def _build_model(arch, cfg):
    if arch == "mlp":
        m = cfg["model"]["mlp"]
        return PhotonEnergyMLP(hidden_dims=m["hidden_dims"], dropout=m["dropout"])
    elif arch == "cnn":
        m = cfg["model"]["cnn"]
        return CalorimeterCNN(channels=m["channels"], dropout=m["dropout"])
    elif arch == "deepsets":
        m = cfg["model"]["deepsets"]
        return DeepSets(phi_dims=m["phi_dims"], rho_dims=m["rho_dims"], dropout=m["dropout"])
    else:
        raise ValueError(f"Unknown arch: {arch}")


def train_single_model(arch, cfg, train_dataset, seed, device, n_epochs=N_EPOCHS):
    """
    Train one ensemble member with a given seed.
    Returns the trained model.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(arch, cfg).to(device)

    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(n_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_ensemble(arch, cfg, train_dataset, ckpt_dir, device):
    """
    Train all ensemble members and save their checkpoints.

    Returns list of trained models.
    """

    seeds   = cfg["uncertainty"]["ensemble_seeds"]
    n_models = cfg["uncertainty"]["ensemble_size"]
    seeds    = seeds[:n_models]

    os.makedirs(ckpt_dir, exist_ok=True)
    models = []

    print(f"\n[Ensemble] Training {n_models} models ...")

    for i, seed in enumerate(seeds):
        print(f"  Model {i+1}/{n_models} (seed={seed}) ...", end=" ", flush=True)

        model = train_single_model(arch, cfg, train_dataset, seed, device)

        # Save individual checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{arch}_ensemble_{i}.pt")
        torch.save({"model_state": model.state_dict(), "seed": seed}, ckpt_path)

        models.append(model)
        print("done")

    return models


def load_ensemble(arch, cfg, ckpt_dir, device):
    """
    Load previously saved ensemble checkpoints.
    """

    n_models = cfg["uncertainty"]["ensemble_size"]
    models   = []

    for i in range(n_models):
        ckpt_path = os.path.join(ckpt_dir, f"{arch}_ensemble_{i}.pt")
        if not os.path.exists(ckpt_path):
            return None    # Signal that ensemble needs to be trained first

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = _build_model(arch, cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)

    return models


def ensemble_predict(models, X_tensor, batch_size=256, device=None):
    """
    Run inference with all ensemble members.
    Returns (means, stds) per sample.
    """

    if device is None:
        device = next(models[0].parameters()).device

    N = X_tensor.shape[0]
    all_preds = np.zeros((len(models), N), dtype=np.float32)

    for k, model in enumerate(models):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = X_tensor[i:i + batch_size].to(device)
                out   = model(batch).cpu().numpy().flatten()
                preds.append(out)
        all_preds[k] = np.concatenate(preds)

    means = all_preds.mean(axis=0)
    stds  = all_preds.std(axis=0)

    return means, stds


def plot_ensemble_uncertainty(y_true, means, stds, plots_dir, mc_stds=None):
    """
    Plot ensemble uncertainty vs energy.
    If mc_stds is provided, add a side-by-side comparison with MC Dropout.
    """

    n_plots = 2 if mc_stds is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Pred vs true
    ax = axes[0]
    ax.scatter(y_true, means, s=3, alpha=0.4, color="forestgreen")
    lo = min(y_true.min(), means.min()) * 0.98
    hi = max(y_true.max(), means.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Ensemble Mean Prediction [GeV]")
    ax.set_title("Deep Ensemble: Predictions")

    # Uncertainty vs energy
    ax = axes[1]
    ax.scatter(y_true, stds, s=3, alpha=0.4, color="forestgreen", label="Ensemble")
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Std across ensemble members [GeV]")
    ax.set_title("Ensemble Epistemic Uncertainty vs Energy")

    # Comparison with MC Dropout (if available)
    if mc_stds is not None:
        ax = axes[2]
        ax.scatter(y_true, mc_stds,  s=3, alpha=0.3, color="steelblue",   label="MC Dropout")
        ax.scatter(y_true, stds,     s=3, alpha=0.3, color="forestgreen",  label="Ensemble")
        ax.set_xlabel("True Energy [GeV]")
        ax.set_ylabel("Uncertainty Std [GeV]")
        ax.set_title("MC Dropout vs Ensemble Uncertainty")
        ax.legend(markerscale=3)

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "ensemble_uncertainty.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  Saved: ensemble_uncertainty.png")


def run(arch, cfg, train_dataset, X_val_tensor, y_true, plots_dir, ckpt_dir, device, mc_stds=None):
    """
    Convenience wrapper: train (or load) ensemble, predict, plot.
    """

    print("\n[Deep Ensemble]")

    # Try loading saved ensemble first
    models = load_ensemble(arch, cfg, ckpt_dir, device)

    if models is None:
        models = train_ensemble(arch, cfg, train_dataset, ckpt_dir, device)

    means, stds = ensemble_predict(models, X_val_tensor, device=device)

    print(f"  Mean uncertainty (std): {stds.mean():.4f} ± {stds.std():.4f} GeV")
    print(f"  Max uncertainty:        {stds.max():.4f} GeV")

    os.makedirs(plots_dir, exist_ok=True)
    plot_ensemble_uncertainty(y_true, means, stds, plots_dir, mc_stds=mc_stds)

    return means, stds
