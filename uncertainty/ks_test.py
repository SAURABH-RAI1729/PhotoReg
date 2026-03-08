import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# ============================================================
# uncertainty/ks_test.py
#
# Test whether the ΔE/E distribution is consistent with a
# Gaussian (normal) distribution.
#
# Two tests:
#   1. Kolmogorov-Smirnov (KS) test
#      Compares empirical CDF to the theoretical Gaussian CDF.
#      Works well for large samples.
#
#   2. Shapiro-Wilk test
#      More powerful for detecting non-normality.
#      Standard in statistics but loses power for N > ~5000,
#      so we subsample if needed.
#
# If the distribution is Gaussian, the Gaussian fit σ from
# gaussian_fit.py is a good summary statistic.
# If it's not Gaussian (e.g. heavy tails from shower leakage),
# p68 is more appropriate than σ.
#
# Diagnostic plot:
#   QQ plot (quantile-quantile) -- points should lie on the
#   diagonal if the distribution is Gaussian.
# ============================================================

SHAPIRO_MAX_N = 5000   # Shapiro-Wilk is slow and saturates for large N


def run_ks_test(delta_e_over_e):
    """
    KS test against a normal distribution with the same mean and std.

    Returns
    -------
    ks_stat : KS statistic
    p_value : p-value
    is_normal : bool, True if p > 0.05
    """

    mu  = np.mean(delta_e_over_e)
    sig = np.std(delta_e_over_e)

    ks_stat, p_value = stats.kstest(
        delta_e_over_e,
        "norm",
        args=(mu, sig)
    )

    is_normal = p_value > 0.05

    return ks_stat, p_value, is_normal


def run_shapiro_test(delta_e_over_e, max_n=SHAPIRO_MAX_N):
    """
    Shapiro-Wilk normality test.
    Subsamples to max_n if the array is larger.

    Returns
    -------
    stat    : test statistic
    p_value : p-value
    is_normal : bool, True if p > 0.05
    n_used  : number of samples used
    """

    rng = np.random.default_rng(42)

    if len(delta_e_over_e) > max_n:
        idx    = rng.choice(len(delta_e_over_e), size=max_n, replace=False)
        sample = delta_e_over_e[idx]
    else:
        sample = delta_e_over_e

    stat, p_value = stats.shapiro(sample)
    is_normal     = p_value > 0.05

    return stat, p_value, is_normal, len(sample)


def plot_ks_diagnostic(delta_e_over_e, plots_dir, filename="ks_diagnostic.png"):
    """
    Two-panel diagnostic plot:
      Left:  empirical CDF vs theoretical Gaussian CDF
      Right: QQ plot
    """

    mu  = np.mean(delta_e_over_e)
    sig = np.std(delta_e_over_e)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ---- CDF comparison ----
    ax = axes[0]

    sorted_data = np.sort(delta_e_over_e * 100)
    n           = len(sorted_data)
    empirical   = np.arange(1, n + 1) / n

    x_theory = np.linspace(sorted_data.min(), sorted_data.max(), 500)
    cdf_theory = stats.norm.cdf(x_theory, loc=mu * 100, scale=sig * 100)

    ax.plot(sorted_data,  empirical,  color="steelblue", linewidth=1.5, label="Empirical CDF")
    ax.plot(x_theory,     cdf_theory, color="red",       linewidth=1.5, linestyle="--",
            label="Gaussian CDF")

    ax.set_xlabel(r"$\Delta E / E$ [%]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical vs Theoretical CDF (KS Test)")
    ax.legend()

    # ---- QQ plot ----
    ax = axes[1]

    rng    = np.random.default_rng(42)
    # subsample for readability
    if len(delta_e_over_e) > 3000:
        idx  = rng.choice(len(delta_e_over_e), size=3000, replace=False)
        sample = delta_e_over_e[idx]
    else:
        sample = delta_e_over_e

    (osm, osr), (slope, intercept, r) = stats.probplot(sample, dist="norm")

    ax.scatter(osm, osr, s=4, alpha=0.4, color="steelblue", label="Data")
    ax.plot(
        [osm.min(), osm.max()],
        [slope * osm.min() + intercept, slope * osm.max() + intercept],
        "r--", linewidth=1.5, label="Normal reference"
    )
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel(r"$\Delta E / E$ quantiles")
    ax.set_title("QQ Plot")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


def run(delta_e_over_e, plots_dir):
    """
    Run both normality tests, print results, and save diagnostic plot.
    """

    print("\n[Normality Tests on ΔE/E Residuals]")

    # KS test
    ks_stat, ks_p, ks_normal = run_ks_test(delta_e_over_e)
    verdict_ks = "consistent with Gaussian ✓" if ks_normal else "NOT consistent with Gaussian ✗"
    print(f"  KS test     :  stat={ks_stat:.4f},  p={ks_p:.4f}  ({verdict_ks})")

    # Shapiro-Wilk
    sw_stat, sw_p, sw_normal, n_used = run_shapiro_test(delta_e_over_e)
    verdict_sw = "consistent with Gaussian ✓" if sw_normal else "NOT consistent with Gaussian ✗"
    print(f"  Shapiro-Wilk:  stat={sw_stat:.4f},  p={sw_p:.4f}  (n={n_used}, {verdict_sw})")

    if not ks_normal:
        print(
            "  Note: Non-Gaussian tails likely reflect energy leakage or "
            "noise floor effects. Use p68 rather than σ as the resolution metric."
        )

    os.makedirs(plots_dir, exist_ok=True)
    plot_ks_diagnostic(delta_e_over_e, plots_dir)

    return {
        "ks_stat"   : ks_stat,
        "ks_p"      : ks_p,
        "ks_normal" : ks_normal,
        "sw_stat"   : sw_stat,
        "sw_p"      : sw_p,
        "sw_normal" : sw_normal
    }
