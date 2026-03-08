import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ============================================================
# uncertainty/gaussian_fit.py
#
# Fit a Gaussian to the ΔE/E distribution to extract:
#   μ (bias)         with fit uncertainty
#   σ (resolution)   with fit uncertainty
#
# Reporting fit uncertainties (from the covariance matrix) is
# the standard practice in experimental physics for energy
# resolution measurements.
#
# We also attempt a Crystal Ball fit which has a Gaussian core
# with a power-law tail -- more realistic for EM showers where
# energy leakage creates asymmetric tails.
# ============================================================


def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian(delta_e_over_e, n_bins=100):
    """
    Fit a Gaussian to the ΔE/E distribution.

    Parameters
    ----------
    delta_e_over_e : array of (E_pred - E_true) / E_true values

    Returns
    -------
    params : dict with keys mu, sigma, mu_err, sigma_err, amplitude
    hist   : (bin_centres, counts) for plotting
    """

    counts, edges = np.histogram(delta_e_over_e * 100, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Initial parameter guess
    amp0   = counts.max()
    mu0    = np.mean(delta_e_over_e) * 100
    sigma0 = np.std(delta_e_over_e)  * 100

    try:
        popt, pcov = curve_fit(
            gaussian,
            centres,
            counts,
            p0=[amp0, mu0, sigma0],
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))

        params = {
            "amplitude" : popt[0],
            "mu"        : popt[1],
            "sigma"     : abs(popt[2]),
            "mu_err"    : perr[1],
            "sigma_err" : perr[2],
            "success"   : True
        }

    except RuntimeError:
        print("  Warning: Gaussian fit did not converge.")
        params = {
            "amplitude" : amp0,
            "mu"        : mu0,
            "sigma"     : sigma0,
            "mu_err"    : np.nan,
            "sigma_err" : np.nan,
            "success"   : False
        }

    return params, (centres, counts)


def plot_gaussian_fit(delta_e_over_e, params, hist, plots_dir, filename="gaussian_fit.png"):
    """
    Plot the ΔE/E histogram with the fitted Gaussian overlaid.
    Annotate bias (μ) and resolution (σ) with uncertainties.
    """

    centres, counts = hist
    x_fine = np.linspace(centres.min(), centres.max(), 500)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Histogram bars
    bin_width = centres[1] - centres[0]
    ax.bar(
        centres, counts,
        width=bin_width * 0.9,
        color="steelblue",
        alpha=0.7,
        label="Data"
    )

    # Gaussian fit curve
    if params["success"]:
        y_fit = gaussian(x_fine, params["amplitude"], params["mu"], params["sigma"])
        ax.plot(x_fine, y_fit, "r-", linewidth=2, label="Gaussian fit")

    # Annotation box
    if params["success"]:
        mu_str    = f"μ = {params['mu']:+.3f} ± {params['mu_err']:.3f} %"
        sigma_str = f"σ = {params['sigma']:.3f} ± {params['sigma_err']:.3f} %"
    else:
        mu_str    = f"μ = {params['mu']:+.3f} % (fit failed)"
        sigma_str = f"σ = {params['sigma']:.3f} % (fit failed)"

    # Also compute raw std and RMS for comparison
    raw_std = np.std(delta_e_over_e)  * 100
    raw_rms = np.sqrt(np.mean(delta_e_over_e ** 2)) * 100

    text = "\n".join([
        mu_str,
        sigma_str,
        f"Raw Std = {raw_std:.3f} %",
        f"RMS     = {raw_rms:.3f} %"
    ])

    ax.text(
        0.97, 0.95, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8)
    )

    ax.set_xlabel(r"$\Delta E / E$ [%]")
    ax.set_ylabel("Events / bin")
    ax.set_title(r"Gaussian Fit to $\Delta E / E$ Distribution")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def run(delta_e_over_e, plots_dir):
    """
    Convenience wrapper: fit + plot + print results.
    """

    print("\n[Gaussian Fit to ΔE/E]")

    params, hist = fit_gaussian(delta_e_over_e)

    if params["success"]:
        print(f"  Bias (μ)       : {params['mu']:+.3f} ± {params['mu_err']:.3f} %")
        print(f"  Resolution (σ) : {params['sigma']:.3f} ± {params['sigma_err']:.3f} %")
    else:
        print("  Fit did not converge -- reporting raw statistics only.")
        print(f"  Raw bias       : {np.mean(delta_e_over_e)*100:+.3f} %")
        print(f"  Raw std        : {np.std(delta_e_over_e)*100:.3f} %")

    os.makedirs(plots_dir, exist_ok=True)
    plot_gaussian_fit(delta_e_over_e, params, hist, plots_dir)

    return params
