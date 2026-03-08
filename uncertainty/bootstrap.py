import numpy as np

# ============================================================
# uncertainty/bootstrap.py
#
# Generic bootstrap confidence interval estimation.
#
# Works with any scalar metric function that takes
# (y_true, y_pred) as arguments.
# ============================================================


def bootstrap_ci(metric_fn, y_true, y_pred, n_iterations=1000, ci=95):
    """
    Estimate a confidence interval for a metric via bootstrap resampling.

    Parameters
    ----------
    metric_fn    : callable(y_true, y_pred) -> float
    y_true       : numpy array
    y_pred       : numpy array
    n_iterations : number of bootstrap resamples
    ci           : confidence level in percent (e.g. 95)

    Returns
    -------
    estimate : metric value on original sample
    lower    : lower CI bound
    upper    : upper CI bound
    """

    rng = np.random.default_rng(42)

    n = len(y_true)
    estimate = metric_fn(y_true, y_pred)

    boot_values = np.zeros(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (100 - ci) / 2.0
    lower = np.percentile(boot_values, alpha)
    upper = np.percentile(boot_values, 100 - alpha)

    return estimate, lower, upper


def bootstrap_ci_array(metric_fn, y_true, y_pred, n_iterations=1000, ci=95):
    """
    Same as bootstrap_ci but for metric functions that return
    an array (e.g. per-bin resolution).

    Returns
    -------
    estimate : array from original sample
    lower    : lower CI bound per element
    upper    : upper CI bound per element
    """

    rng = np.random.default_rng(42)

    n        = len(y_true)
    estimate = metric_fn(y_true, y_pred)
    k        = len(estimate)

    boot_values = np.zeros((n_iterations, k))

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (100 - ci) / 2.0
    lower = np.percentile(boot_values, alpha,       axis=0)
    upper = np.percentile(boot_values, 100 - alpha, axis=0)

    return estimate, lower, upper


def report_with_ci(name, estimate, lower, upper, unit=""):
    """
    Pretty-print a metric with its confidence interval.
    """
    print(f"  {name:<28}  {estimate:+.3f}  [{lower:+.3f}, {upper:+.3f}]  {unit}")
