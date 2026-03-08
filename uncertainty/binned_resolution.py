import numpy as np
import matplotlib.pyplot as plt
import os

from uncertainty.bootstrap import bootstrap_ci

# ============================================================
# uncertainty/binned_resolution.py
#
# Compute and plot energy resolution (p68 of |ΔE/E|) in bins
# of true photon energy, with bootstrap uncertainty bars.
#
# This is the standard physics plot for evaluating calorimeter
# performance as a function of energy.
# ============================================================


def compute_binned_resolution(y_true, y_pred, n_bins=10, n_boot=1000, ci=95):
    """
    Compute p68(|ΔE/E|) in equal-statistics bins of y_true.

    Using equal-statistics bins (percentile-based) ensures each
    bin has enough events for a reliable bootstrap estimate.

    Returns
    -------
    bin_centres  : array (n_bins,)
    resolutions  : array (n_bins,) -- p68 in percent
    lower_errors : array (n_bins,) -- lower CI bound in percent
    upper_errors : array (n_bins,) -- upper CI bound in percent
    """

    bin_edges   = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_centres = []
    resolutions = []
    lower_errs  = []
    upper_errs  = []

    for i in range(n_bins):

        lo   = bin_edges[i]
        hi   = bin_edges[i + 1]
        mask = (y_true >= lo) & (y_true < hi)

        if mask.sum() < 20:
            # Skip bins with very few events
            continue

        yt_bin = y_true[mask]
        yp_bin = y_pred[mask]

        # p68 for this energy bin
        def p68_metric(yt, yp):
            dE_E = (yp - yt) / yt
            return np.percentile(np.abs(dE_E), 68) * 100.0

        est, lo_ci, hi_ci = bootstrap_ci(
            p68_metric, yt_bin, yp_bin,
            n_iterations=n_boot,
            ci=ci
        )

        bin_centres.append(0.5 * (lo + hi))
        resolutions.append(est)
        lower_errs.append(est - lo_ci)
        upper_errs.append(hi_ci - est)

    return (
        np.array(bin_centres),
        np.array(resolutions),
        np.array(lower_errs),
        np.array(upper_errs)
    )


def plot_binned_resolution(
    bin_centres, resolutions, lower_errs, upper_errs,
    plots_dir, filename="binned_resolution.png"
):
    """
    Plot p68 resolution vs true energy with bootstrap error bars.
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        bin_centres,
        resolutions,
        yerr=[lower_errs, upper_errs],
        fmt="o-",
        color="steelblue",
        capsize=5,
        capthick=1.2,
        elinewidth=1.2,
        label=r"$p_{68}(|\Delta E/E|)$"
    )

    ax.set_xlabel("True Photon Energy [GeV]")
    ax.set_ylabel(r"$p_{68}(|\Delta E / E|)$ [%]")
    ax.set_title("Binned Energy Resolution with Bootstrap Uncertainties")
    ax.legend()

    fig.tight_layout()

    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  Saved: {filename}")


def run(y_true, y_pred, plots_dir, n_bins=10):
    """
    Convenience wrapper: compute + plot.
    """

    print("\n[Binned Resolution]")

    centres, res, lo_err, hi_err = compute_binned_resolution(
        y_true, y_pred, n_bins=n_bins
    )

    os.makedirs(plots_dir, exist_ok=True)
    plot_binned_resolution(centres, res, lo_err, hi_err, plots_dir)

    print(f"  Energy range: [{centres.min():.1f}, {centres.max():.1f}] GeV")
    print(f"  Resolution range: [{res.min():.2f}%, {res.max():.2f}%]")

    return centres, res, lo_err, hi_err
