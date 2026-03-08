import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml

# Local imports
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset  import load_data, get_dataloaders, save_scaler
from models.mlp    import PhotonEnergyMLP, count_parameters
from models.cnn    import CalorimeterCNN
from models.deepsets import DeepSets

# ============================================================
# train.py
#
# Main training script.
#
# Usage:
#   python train.py --config configs/config.yaml
#
# Saves:
#   checkpoints/<arch>_best.pt   -- best validation loss
#   checkpoints/<arch>_last.pt   -- final epoch
#   checkpoints/scaler.pkl       -- fitted StandardScaler
#   plots/loss_curve.png         -- full loss curve
#   plots/loss_curve_clipped.png -- same but epoch 0 removed
#                                   (useful if first epoch loss spikes)
# ============================================================


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    arch = cfg["model"]["architecture"]

    if arch == "mlp":
        m_cfg = cfg["model"]["mlp"]
        return PhotonEnergyMLP(
            hidden_dims=m_cfg["hidden_dims"],
            dropout=m_cfg["dropout"]
        ), "flat"

    elif arch == "cnn":
        m_cfg = cfg["model"]["cnn"]
        return CalorimeterCNN(
            channels=m_cfg["channels"],
            dropout=m_cfg["dropout"]
        ), "image"

    elif arch == "deepsets":
        m_cfg = cfg["model"]["deepsets"]
        return DeepSets(
            phi_dims=m_cfg["phi_dims"],
            rho_dims=m_cfg["rho_dims"],
            dropout=m_cfg["dropout"]
        ), "deepsets"

    else:
        raise ValueError(f"Unknown architecture: {arch}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


def save_loss_plots(train_losses, val_losses, plots_dir, clip_first):
    """
    Save two loss curve plots:
      1. Full curve including epoch 0
      2. Clipped curve starting from epoch 1 (removes large first-epoch spike)

    The clipped version is only saved if the first epoch loss is notably
    larger than the rest -- useful for diagnosing training dynamics.
    """

    os.makedirs(plots_dir, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    # ---- full plot ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, val_losses,   label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # ---- clipped plot (skip epoch 1 if it's a big spike) ----
    if clip_first and len(train_losses) > 1:
        ratio = train_losses[0] / (train_losses[1] + 1e-9)
        if ratio > 3.0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs[1:], train_losses[1:], label="Train")
            ax.plot(epochs[1:], val_losses[1:],   label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title("Training and Validation Loss (epoch 1 removed)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                os.path.join(plots_dir, "loss_curve_clipped.png"), dpi=150
            )
            plt.close(fig)
            print(
                f"  First epoch loss was {ratio:.1f}x higher than epoch 2 -- "
                "saved clipped loss curve too."
            )


def main(config_path):

    cfg = load_config(config_path)

    # Reproducibility
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    arch = cfg["model"]["architecture"]

    mode_map = {"mlp": "flat", "cnn": "image", "deepsets": "deepsets"}

    train_ds, val_ds, scaler = load_data(
        data_path  = cfg["data"]["path"],
        val_split  = cfg["data"]["val_split"],
        seed       = seed,
        mode       = mode_map[arch]
    )

    train_loader, val_loader = get_dataloaders(
        train_ds, val_ds,
        batch_size = cfg["training"]["batch_size"]
    )

    # Save scaler so inference / uncertainty scripts can use it
    ckpt_dir = cfg["checkpoints"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    save_scaler(scaler, os.path.join(ckpt_dir, "scaler.pkl"))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, _ = build_model(cfg)
    model     = model.to(device)

    n_params = count_parameters(model)
    print(f"\nModel: {arch.upper()}")
    print(f"Trainable parameters: {n_params:,}")

    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    t_cfg     = cfg["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr           = t_cfg["learning_rate"],
        weight_decay = t_cfg["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = cfg["training"]["scheduler"]["patience"],
        factor   = cfg["training"]["scheduler"]["factor"]
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    n_epochs    = t_cfg["epochs"]
    train_losses = []
    val_losses   = []
    best_val     = float("inf")

    print(f"\nTraining for {n_epochs} epochs ...\n")

    for epoch in range(n_epochs):

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:03d}/{n_epochs} | "
            f"Train: {train_loss:.5f} | "
            f"Val:   {val_loss:.5f}"
        )

        # Save best checkpoint
        if cfg["checkpoints"]["save_best"] and val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(ckpt_dir, f"{arch}_best.pt")
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "config"     : cfg
            }, ckpt_path)

    # Save last checkpoint
    if cfg["checkpoints"]["save_last"]:
        ckpt_path = os.path.join(ckpt_dir, f"{arch}_last.pt")
        torch.save({
            "epoch"      : n_epochs,
            "model_state": model.state_dict(),
            "val_loss"   : val_losses[-1],
            "config"     : cfg
        }, ckpt_path)

    print(f"\nBest validation loss: {best_val:.5f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")

    # --------------------------------------------------------
    # Loss plots
    # --------------------------------------------------------

    save_loss_plots(
        train_losses, val_losses,
        plots_dir  = cfg["plots"]["dir"],
        clip_first = cfg["plots"]["loss_clip_first_epoch"]
    )

    print(f"Loss plots saved to: {cfg['plots']['dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML"
    )
    args = parser.parse_args()
    main(args.config)
