import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset    import load_data, load_scaler
from models.mlp      import PhotonEnergyMLP
from models.cnn      import CalorimeterCNN
from models.deepsets import DeepSets
from utils.metrics   import (
    compute_all_metrics, print_metrics, relative_error
)

# ============================================================
# evaluate.py
#
# Loads the best saved checkpoint, runs inference on the
# validation set, computes all metrics, and saves four plots:
#
#   1. Predicted vs True energy
#   2. Histogram of ΔE/E
#   3. |ΔE|/E vs E_true (resolution vs energy)
#   4. Training curves (re-plotted from saved losses if desired,
#      but those are already saved by train.py)
#
# Usage:
#   python evaluate.py --config configs/config.yaml
# ============================================================


def load_model_from_checkpoint(ckpt_path, device):
    """
    Rebuild model from the saved config inside the checkpoint
    and load the saved weights.
    """

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    arch = cfg["model"]["architecture"]

    if arch == "mlp":
        m_cfg = cfg["model"]["mlp"]
        model = PhotonEnergyMLP(
            hidden_dims = m_cfg["hidden_dims"],
            dropout     = m_cfg["dropout"]
        )

    elif arch == "cnn":
        m_cfg = cfg["model"]["cnn"]
        model = CalorimeterCNN(
            channels = m_cfg["channels"],
            dropout  = m_cfg["dropout"]
        )

    elif arch == "deepsets":
        m_cfg = cfg["model"]["deepsets"]
        model = DeepSets(
            phi_dims = m_cfg["phi_dims"],
            rho_dims = m_cfg["rho_dims"],
            dropout  = m_cfg["dropout"]
        )

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, cfg, arch


def get_predictions(model, val_dataset, device, batch_size=256):
    """
    Run inference over the validation dataset.
    Returns numpy arrays: y_true, y_pred.
    """

    from torch.utils.data import DataLoader

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            pred    = model(batch_x).cpu().numpy().flatten()
            y_pred_list.append(pred)
            y_true_list.append(batch_y.numpy().flatten())

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


# ============================================================
# Plotting functions
# ============================================================

def plot_pred_vs_true(y_true, y_pred, plots_dir):
    """
    Scatter plot of predicted vs true energy with the ideal diagonal.
    Points are coloured by density using a 2D histogram.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    h = ax.hist2d(y_true, y_pred, bins=80, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="Counts")

    lo = min(y_true.min(), y_pred.min()) * 0.98
    hi = max(y_true.max(), y_pred.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Ideal")

    ax.set_xlabel("True Photon Energy [GeV]")
    ax.set_ylabel("Predicted Photon Energy [GeV]")
    ax.set_title("Predicted vs True Energy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pred_vs_true.png"), dpi=150)
    plt.close(fig)
    print("  Saved: pred_vs_true.png")


def plot_delta_e_hist(delta_e_over_e, plots_dir):
    """
    Histogram of ΔE/E with mean and RMS annotated.
    """

    mean_val = np.mean(delta_e_over_e) * 100
    rms_val  = np.sqrt(np.mean(delta_e_over_e ** 2)) * 100
    std_val  = np.std(delta_e_over_e) * 100
    p68_val  = np.percentile(np.abs(delta_e_over_e), 68) * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        delta_e_over_e * 100,
        bins=100,
        color="steelblue",
        edgecolor="none",
        alpha=0.8
    )

    ax.axvline(mean_val, color="red",    linestyle="--", label=f"Bias = {mean_val:.2f}%")
    ax.axvline(0,        color="black",  linestyle=":",  linewidth=0.8)

    text = (
        f"RMS = {rms_val:.2f}%\n"
        f"Std = {std_val:.2f}%\n"
        f"p68 = {p68_val:.2f}%"
    )
    ax.text(
        0.97, 0.95, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    ax.set_xlabel(r"$\Delta E / E$ [%]")
    ax.set_ylabel("Events")
    ax.set_title(r"Energy Resolution: $\Delta E / E$ Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "delta_e_hist.png"), dpi=150)
    plt.close(fig)
    print("  Saved: delta_e_hist.png")


def plot_resolution_vs_energy(y_true, delta_e_over_e, plots_dir, n_bins=10):
    """
    |ΔE|/E (p68) in energy bins vs E_true.
    Shows how resolution varies across the energy range.
    """

    bins     = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_centres = []
    resolutions = []
    errors      = []

    for i in range(n_bins):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() < 10:
            continue

        dE_bin = delta_e_over_e[mask]
        p68    = np.percentile(np.abs(dE_bin), 68) * 100

        # Bootstrap uncertainty on p68 for this bin
        boot = [
            np.percentile(
                np.abs(np.random.choice(dE_bin, size=len(dE_bin), replace=True)),
                68
            ) * 100
            for _ in range(500)
        ]
        err = np.std(boot)

        bin_centres.append(0.5 * (bins[i] + bins[i + 1]))
        resolutions.append(p68)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        bin_centres, resolutions, yerr=errors,
        fmt="o-", color="steelblue", capsize=4
    )

    ax.set_xlabel("True Photon Energy [GeV]")
    ax.set_ylabel(r"$p_{68}(|\Delta E / E|)$ [%]")
    ax.set_title("Energy Resolution vs True Energy")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "resolution_vs_energy.png"), dpi=150)
    plt.close(fig)
    print("  Saved: resolution_vs_energy.png")


# ============================================================
# Main
# ============================================================

def main(config_path):

    cfg = yaml.safe_load(open(config_path))

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch     = cfg["model"]["architecture"]
    ckpt_dir = cfg["checkpoints"]["dir"]
    plots_dir = cfg["plots"]["dir"]

    os.makedirs(plots_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    ckpt_path = os.path.join(ckpt_dir, f"{arch}_best.pt")
    print(f"Loading checkpoint: {ckpt_path}")

    model, ckpt_cfg, arch = load_model_from_checkpoint(ckpt_path, device)

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    mode_map = {"mlp": "flat", "cnn": "image", "deepsets": "deepsets"}

    _, val_ds, _ = load_data(
        data_path = cfg["data"]["path"],
        val_split = cfg["data"]["val_split"],
        seed      = seed,
        mode      = mode_map[arch]
    )

    # --------------------------------------------------------
    # Get predictions
    # --------------------------------------------------------

    y_true, y_pred = get_predictions(model, val_ds, device)

    dE_E = relative_error(y_true, y_pred)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_all_metrics(y_true, y_pred)
    print_metrics(metrics)

    # Save metrics to a text file for the README
    with open(os.path.join(plots_dir, "metrics.txt"), "w") as f:
        f.write(f"Architecture: {arch.upper()}\n\n")
        for k, v in metrics.items():
            f.write(f"  {k:<22} {v:.4f}\n")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print("\nSaving plots ...")
    plot_pred_vs_true(y_true, y_pred, plots_dir)
    plot_delta_e_hist(dE_E, plots_dir)
    plot_resolution_vs_energy(y_true, dE_E, plots_dir, n_bins=10)

    print(f"\nAll plots saved to: {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
