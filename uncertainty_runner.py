# ============================================================
# uncertainty_runner.py
#
# Master script that runs all uncertainty analyses in sequence.
#
# Usage:
#   python uncertainty_runner.py --config configs/config.yaml
#
# Outputs (saved to plots/uncertainty/):
#   binned_resolution.png
#   mc_dropout_uncertainty.png
#   gaussian_fit.png
#   ensemble_uncertainty.png   (includes MC vs Ensemble comparison)
#   ks_diagnostic.png
#   uncertainty_summary.txt    (printed table + saved)
# ============================================================


# UNCT_PLOTS_DIR is derived from config inside main()

import os
import sys
import argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset  import load_data
from evaluate      import load_model_from_checkpoint, get_predictions
from utils.metrics import relative_error, bias, rms_resolution, std_resolution, p68_resolution

import uncertainty.bootstrap         as bootstrap_mod
import uncertainty.binned_resolution as binned_res_mod
import uncertainty.mc_dropout        as mc_dropout_mod
import uncertainty.gaussian_fit      as gauss_fit_mod
import uncertainty.ensemble          as ensemble_mod
import uncertainty.ks_test           as ks_mod

from utils.metrics import mae, mse, rmse

def print_summary_table(results):
    """
    Print a formatted summary of all uncertainty results.
    """

    sep = "=" * 66

    print("\n" + sep)
    print(f"{'UNCERTAINTY QUANTIFICATION SUMMARY':^66}")
    print(sep)
    print(f"  {'Metric':<35}  {'Value':>10}  {'95% CI / Note'}")
    print("-" * 66)

    for row in results:
        name, value, note = row
        print(f"  {name:<35}  {value:>10}  {note}")

    print(sep + "\n")


def main(config_path):

    cfg  = yaml.safe_load(open(config_path))
    seed = cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch   = cfg["model"]["architecture"]

    global UNCT_PLOTS_DIR
    UNCT_PLOTS_DIR = os.path.join(cfg["plots"]["dir"], "uncertainty")
    os.makedirs(UNCT_PLOTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load model and get predictions
    # --------------------------------------------------------

    ckpt_path = os.path.join(cfg["checkpoints"]["dir"], f"{arch}_best.pt")
    print(f"Loading model: {ckpt_path}")

    model, ckpt_cfg, arch = load_model_from_checkpoint(ckpt_path, device)

    mode_map = {"mlp": "flat", "cnn": "image", "deepsets": "deepsets"}

    train_ds, val_ds, _ = load_data(
        data_path = cfg["data"]["path"],
        val_split = cfg["data"]["val_split"],
        seed      = seed,
        mode      = mode_map[arch]
    )

    y_true, y_pred = get_predictions(model, val_ds, device)
    dE_E           = relative_error(y_true, y_pred)

    X_val_tensor   = val_ds.X   # needed for MC Dropout and Ensemble

    # --------------------------------------------------------
    # 2. Bootstrap CIs on standard metrics
    # --------------------------------------------------------

    print("\n[Bootstrap CIs on Metrics]")

    unct_cfg = cfg["uncertainty"]
    n_boot   = unct_cfg["bootstrap_iterations"]
    ci_level = unct_cfg["bootstrap_ci"]

    def _bias_pct(yt, yp):
        return np.mean((yp - yt) / yt) * 100

    def _rms_pct(yt, yp):
        dE = (yp - yt) / yt
        return np.sqrt(np.mean(dE ** 2)) * 100

    def _std_pct(yt, yp):
        return np.std((yp - yt) / yt) * 100

    def _p68_pct(yt, yp):
        return np.percentile(np.abs((yp - yt) / yt), 68) * 100

    def _mae_fn(yt, yp):
        return np.mean(np.abs(yp - yt))

    def _rmse_fn(yt, yp):
        return np.sqrt(np.mean((yp - yt) ** 2))

    boot_results = {}

    for name, fn in [
        ("Bias [%]",        _bias_pct),
        ("RMS(dE/E) [%]",   _rms_pct),
        ("Std(dE/E) [%]",   _std_pct),
        ("p68(|dE/E|) [%]", _p68_pct),
        ("MAE [GeV]",       _mae_fn),
        ("RMSE [GeV]",      _rmse_fn),
    ]:
        est, lo, hi = bootstrap_mod.bootstrap_ci(
            fn, y_true, y_pred,
            n_iterations=n_boot,
            ci=ci_level
        )
        boot_results[name] = (est, lo, hi)
        bootstrap_mod.report_with_ci(name, est, lo, hi, f"({ci_level}% CI)")

    # --------------------------------------------------------
    # 3. Binned resolution
    # --------------------------------------------------------

    n_bins = unct_cfg["n_bins_resolution"]
    centres, res, lo_err, hi_err = binned_res_mod.run(
        y_true, y_pred, UNCT_PLOTS_DIR, n_bins=n_bins
    )

    # --------------------------------------------------------
    # 4. MC Dropout
    # --------------------------------------------------------

    n_passes = unct_cfg["mc_dropout_passes"]
    mc_means, mc_stds = mc_dropout_mod.run(
        model, X_val_tensor, y_true, UNCT_PLOTS_DIR, n_passes=n_passes
    )

    # --------------------------------------------------------
    # 5. Gaussian fit to ΔE/E
    # --------------------------------------------------------

    gauss_params = gauss_fit_mod.run(dE_E, UNCT_PLOTS_DIR)

    # --------------------------------------------------------
    # 6. Deep Ensemble
    # --------------------------------------------------------

    ens_means, ens_stds = ensemble_mod.run(
        arch        = arch,
        cfg         = cfg,
        train_dataset = train_ds,
        X_val_tensor  = X_val_tensor,
        y_true        = y_true,
        plots_dir     = UNCT_PLOTS_DIR,
        ckpt_dir      = cfg["checkpoints"]["dir"],
        device        = device,
        mc_stds       = mc_stds       # passes MC stds for comparison plot
    )

    # --------------------------------------------------------
    # 7. KS test
    # --------------------------------------------------------

    ks_results = ks_mod.run(dE_E, UNCT_PLOTS_DIR)

    # --------------------------------------------------------
    # 8. Print and save summary table
    # --------------------------------------------------------

    summary_rows = []

    # Bootstrap metrics
    for name, (est, lo, hi) in boot_results.items():
        summary_rows.append((
            name,
            est,
            f"[{lo:+.3f}, {hi:+.3f}]  Bootstrap {ci_level}% CI"
        ))

    # Gaussian fit
    if gauss_params["success"]:
        summary_rows.append((
            "Gaussian fit μ [%]",
            gauss_params["mu"],
            f"± {gauss_params['mu_err']:.3f}  Fit uncertainty"
        ))
        summary_rows.append((
            "Gaussian fit σ [%]",
            gauss_params["sigma"],
            f"± {gauss_params['sigma_err']:.3f}  Fit uncertainty"
        ))

    # MC Dropout
    summary_rows.append((
        "MC Dropout mean σ [GeV]",
        mc_stds.mean(),
        f"± {mc_stds.std():.4f}"
    ))

    # Ensemble
    summary_rows.append((
        "Ensemble mean σ [GeV]",
        ens_stds.mean(),
        f"± {ens_stds.std():.4f}"
    ))

    # KS test
    ks_verdict = "Normal ✓" if ks_results["ks_normal"] else "Non-normal ✗"
    summary_rows.append((
        "KS statistic",
        ks_results["ks_stat"],
        f"p={ks_results['ks_p']:.4f}  {ks_verdict}"
    ))
    sw_verdict = "Normal ✓" if ks_results["sw_normal"] else "Non-normal ✗"
    summary_rows.append((
        "Shapiro-Wilk statistic",
        ks_results["sw_stat"],
        f"p={ks_results['sw_p']:.4f}  {sw_verdict}"
    ))

    print_summary_table(summary_rows)

    # Save to file
    out_path = os.path.join(UNCT_PLOTS_DIR, "uncertainty_summary.txt")
    with open(out_path, "w") as f:
        f.write(f"Architecture: {arch.upper()}\n\n")
        f.write(f"{'Metric':<35}  {'Value':>10}  Note\n")
        f.write("-" * 70 + "\n")
        for name, value, note in summary_rows:
            f.write(f"{name:<35}  {value:>10.4f}  {note}\n")

    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)