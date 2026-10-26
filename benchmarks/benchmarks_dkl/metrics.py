# standard imports
from typing import Optional, Tuple

# package imports
import numpy as np
from scipy.stats import norm as _scipy_norm
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)

"""
Metrics for evaluating models. All calculations are carried out using numpy
arrays as the base data type.
"""


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_levels: int = 99,
    mode: str = "interval",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a probabilistic calibration curve for a Gaussian predictive model.

    For each confidence level alpha in a uniform grid from 1/(n_levels+1)
    to n_levels/(n_levels+1), the empirical coverage is measured and compared
    to the expected alpha. A perfectly calibrated model lies on the diagonal.

    Two modes are supported:

    * "interval" *(default)* — symmetric interval calibration.
      For each alpha, checks whether |y_true - y_pred| <= z_alpha * y_std
      where z_alpha = Phi^{-1}((1 + alpha) / 2).  Equivalent to asking
      "does the symmetric [y_pred ± z*sigma] band capture alpha of the data?"

    * "quantile" — CDF / quantile calibration.
      Normalises residuals as r = (y_true - y_pred) / y_std and checks
      whether each r falls below the alpha-quantile of the standard normal,
      i.e. r < Phi^{-1}(alpha).  Equivalent to asking "does the predictive
      CDF assign the correct probability to the left tail?"  This mode can
      detect asymmetric bias that interval calibration misses.

    In both cases:

    * Points **above the diagonal** → under-confident (intervals/tails too wide).
    * Points **below the diagonal** → over-confident (intervals/tails too narrow).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Predicted mean values.
    y_std : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Predicted standard deviations.
    n_levels : int, default 99
        Number of confidence levels to evaluate. Levels are uniformly spaced
        between 0 (exclusive) and 1 (exclusive).
    mode : {"interval", "quantile"}, default "interval"
        Calibration mode. See above for details.

    Returns
    -------
    expected : np.ndarray of shape (n_levels,)
        Confidence levels (x-axis of the calibration plot).
    observed : np.ndarray of shape (n_levels,)
        Empirical coverage at each confidence level (y-axis).

    Raises
    ------
    ValueError
        If mode is not "interval" or "quantile".
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_std = np.asarray(y_std).ravel()

    expected = np.linspace(1 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)

    if mode == "interval":
        z = _scipy_norm.ppf((1 + expected) / 2)
        residuals = np.abs(y_true - y_pred)
        observed = np.array([np.mean(residuals <= z_i * y_std) for z_i in z])
    elif mode == "quantile":
        residuals = (y_true - y_pred) / y_std
        observed = np.array([np.mean(residuals <= _scipy_norm.ppf(alpha)) for alpha in expected])
    else:
        raise ValueError(f"mode must be 'interval' or 'quantile', got {mode!r}")

    return expected, observed


def calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_levels: int = 99,
    mode: str = "interval",
    multioutput: str = "uniform_average",
) -> float:
    r"""
    Mean absolute calibration error (MACE).

    Computes the mean absolute deviation between the empirical coverage and the
    expected coverage across n_levels uniformly spaced confidence levels:

        MACE = 1/K \sum_{k=1}^{K} |\hat{p}(\alpha_k) - \alpha_k|

    A perfectly calibrated model has MACE = 0. Smaller is better.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Ground-truth target values.
    y_pred : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Predicted mean values.
    y_std : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Predicted standard deviations.
    n_levels : int, default 99
        Number of confidence levels passed to calibration_curve.
    mode : {"interval", "quantile"}, default "interval"
        Calibration mode passed to calibration_curve. "interval"
        checks symmetric interval coverage; "quantile" checks CDF
        calibration and can detect asymmetric bias.
    multioutput : str, default 'uniform_average'
        Ignored (accepted for API consistency with other metrics).

    Returns
    -------
    float
        Mean absolute calibration error.
    """
    expected, observed = calibration_curve(y_true, y_pred, y_std, n_levels=n_levels, mode=mode)
    return float(np.mean(np.abs(observed - expected)))
