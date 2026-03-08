import numpy as np

# ============================================================
# utils/metrics.py
#
# Standard regression metrics + physics-specific energy
# resolution metrics.
#
# All functions take plain numpy arrays.
# ============================================================


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def relative_error(y_true, y_pred):
    """
    ΔE / E = (E_pred - E_true) / E_true

    Returns array of same length as inputs.
    """
    return (y_pred - y_true) / y_true


def bias(delta_e_over_e):
    """
    Mean of ΔE/E -- measures systematic shift.
    Reported in percent.
    """
    return np.mean(delta_e_over_e) * 100.0


def rms_resolution(delta_e_over_e):
    """
    RMS(ΔE/E) -- total spread including tails.
    Reported in percent.
    """
    return np.sqrt(np.mean(delta_e_over_e ** 2)) * 100.0


def std_resolution(delta_e_over_e):
    """
    Standard deviation of ΔE/E.
    Reported in percent.
    """
    return np.std(delta_e_over_e) * 100.0


def p68_resolution(delta_e_over_e):
    """
    68th percentile of |ΔE/E|.

    This is the standard physics metric for energy resolution --
    analogous to the 1-sigma interval for a Gaussian without
    assuming the distribution is Gaussian.
    Reported in percent.
    """
    return np.percentile(np.abs(delta_e_over_e), 68) * 100.0


def compute_all_metrics(y_true, y_pred):
    """
    Convenience wrapper: compute and return all metrics as a dict.
    """

    dE_E = relative_error(y_true, y_pred)

    return {
        "MAE"            : mae(y_true, y_pred),
        "MSE"            : mse(y_true, y_pred),
        "RMSE"           : rmse(y_true, y_pred),
        "Bias [%]"       : bias(dE_E),
        "RMS(dE/E) [%]"  : rms_resolution(dE_E),
        "Std(dE/E) [%]"  : std_resolution(dE_E),
        "p68(|dE/E|) [%]": p68_resolution(dE_E),
    }


def print_metrics(metrics_dict):
    """
    Pretty-print a metrics dictionary.
    """
    print("\n" + "=" * 45)
    print(f"{'Metric':<22}  {'Value':>10}")
    print("-" * 45)
    for k, v in metrics_dict.items():
        print(f"  {k:<20}  {v:>10.4f}")
    print("=" * 45)
