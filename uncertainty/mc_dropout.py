import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# ============================================================
# uncertainty/mc_dropout.py
#
# Monte Carlo Dropout uncertainty estimation.
#
# Key idea:
#   Normally dropout is turned OFF at inference (model.eval()).
#   Here we deliberately keep dropout ON and run N_PASSES forward
#   passes per sample. The spread across passes estimates the
#   model's epistemic (parameter) uncertainty.
#
# This only makes sense if the model was trained with Dropout
# layers -- our MLP, CNN, and DeepSets all include Dropout.
#
# Reference:
#   Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
# ============================================================


def _enable_dropout(model):
    """
    Set all Dropout layers to training mode so they remain active
    during inference. All other layers (BatchNorm etc.) stay in
    eval mode for stable statistics.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(model, X_tensor, n_passes=100, batch_size=256, device=None):
    """
    Run N_PASSES stochastic forward passes and return mean and std
    of the predictions for each sample.

    Parameters
    ----------
    model    : trained PyTorch model (must have Dropout layers)
    X_tensor : torch.Tensor of shape (N, ...) -- already on CPU is fine
    n_passes : number of stochastic forward passes
    batch_size : batch size for inference

    Returns
    -------
    means : numpy array (N,) -- mean prediction across passes
    stds  : numpy array (N,) -- std across passes (epistemic uncertainty)
    """

    if device is None:
        device = next(model.parameters()).device

    # Put model in eval mode first (freezes BN statistics)
    model.eval()
    # Then selectively re-enable dropout
    _enable_dropout(model)

    N = X_tensor.shape[0]
    all_preds = np.zeros((n_passes, N), dtype=np.float32)

    with torch.no_grad():
        for p in range(n_passes):
            preds = []
            for i in range(0, N, batch_size):
                batch = X_tensor[i:i + batch_size].to(device)
                out   = model(batch).cpu().numpy().flatten()
                preds.append(out)
            all_preds[p] = np.concatenate(preds)

    means = all_preds.mean(axis=0)
    stds  = all_preds.std(axis=0)

    # Restore model to full eval mode
    model.eval()

    return means, stds


def plot_uncertainty_vs_energy(y_true, means, stds, plots_dir, filename="mc_dropout_uncertainty.png"):
    """
    Scatter plot of per-sample MC Dropout uncertainty (std) vs true energy.
    Helps show whether the model is more uncertain at certain energies.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: predicted vs true with uncertainty shading
    order = np.argsort(y_true)
    ax = axes[0]
    ax.scatter(y_true, means, s=3, alpha=0.4, color="steelblue", label="MC mean")
    lo = means - stds
    hi = means + stds
    # Only draw band for a random subset to keep the plot readable
    sub = np.random.choice(len(y_true), size=min(2000, len(y_true)), replace=False)
    ax.errorbar(
        y_true[sub], means[sub],
        yerr=stds[sub],
        fmt="none",
        alpha=0.2,
        color="steelblue",
        elinewidth=0.5
    )
    lo_diag = min(y_true.min(), means.min()) * 0.98
    hi_diag = max(y_true.max(), means.max()) * 1.02
    ax.plot([lo_diag, hi_diag], [lo_diag, hi_diag], "r--", linewidth=1)
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("MC Predicted Energy [GeV]")
    ax.set_title("MC Dropout: Predictions with Uncertainty")
    ax.legend(markerscale=3)

    # Right: uncertainty (std) vs true energy
    ax = axes[1]
    ax.scatter(y_true, stds, s=3, alpha=0.4, color="darkorange")
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("MC Dropout Std [GeV]")
    ax.set_title("Per-Sample Epistemic Uncertainty vs Energy")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def run(model, X_tensor, y_true, plots_dir, n_passes=100):
    """
    Convenience wrapper: run MC Dropout and save plot.
    """

    print(f"\n[MC Dropout] Running {n_passes} forward passes ...")

    means, stds = mc_dropout_predict(model, X_tensor, n_passes=n_passes)

    print(f"  Mean uncertainty (std): {stds.mean():.4f} ± {stds.std():.4f} GeV")
    print(f"  Max uncertainty:        {stds.max():.4f} GeV")

    os.makedirs(plots_dir, exist_ok=True)
    plot_uncertainty_vs_energy(y_true, means, stds, plots_dir)

    return means, stds
