"""
Synthetic LPPLS data generation (spec §7).

Functions
---------
generate_lppls_series   – clean LPPLS series rescaled to [0,1]
add_white_noise         – white Gaussian noise augmentation (§7.2)
add_ar1_noise           – AR(1) noise augmentation (§7.3)
sample_lppls_params     – sample random {tc, m, omega} within spec ranges
"""

import numpy as np
from scipy.signal import lfilter

from .formula import lppls_reformulated


def generate_lppls_series(
    tc: float, m: float, omega: float,
    A: float, B: float, C1: float, C2: float,
    n: int, t2: float = 1.0
) -> np.ndarray:
    """Generate a clean LPPLS series and rescale to [0, 1] (spec §7.1).

    Parameters
    ----------
    tc : float
        Critical time (normalised units, > t2 for a post-series singularity).
    m : float
        Power-law exponent.
    omega : float
        Log-periodic angular frequency.
    A, B, C1, C2 : float
        Linear LPPLS parameters.
    n : int
        Number of time points.
    t2 : float
        End of time series (normalised to 1.0 by convention, spec §2).

    Returns
    -------
    np.ndarray of shape (n,)
        Min-max scaled to [0, 1].
    """
    t = np.linspace(0.0, t2, n)
    series = lppls_reformulated(t, tc, m, omega, A, B, C1, C2)

    s_min = series.min()
    s_max = series.max()
    if s_max - s_min < 1e-12:
        return np.zeros(n)
    return (series - s_min) / (s_max - s_min)


def add_white_noise(
    series: np.ndarray,
    alpha: float,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add white Gaussian noise to a clean series (spec §7.2).

    s'_i = s_i + eta_i,   eta_i ~ N(0, alpha^2)

    Parameters
    ----------
    series : np.ndarray
        Clean series (values in [0, 1]).
    alpha : float
        Noise standard deviation (fraction of function range; range is 1 for
        a [0,1]-scaled series).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray, same shape as series.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=alpha, size=series.shape)
    return series + noise


def add_ar1_noise(
    series: np.ndarray,
    phi_ar: float = 0.9,
    sigma: float = 0.02,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add AR(1) noise to a clean series (spec §7.3).

    eta_t = phi_ar * eta_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)
    AR(1) variance: sigma^2 / (1 - phi_ar^2)

    Parameters
    ----------
    series : np.ndarray
        Clean series.
    phi_ar : float
        AR(1) coefficient (spec: 0.9).
    sigma : float
        Innovation standard deviation.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray, same shape as series.
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = rng.normal(loc=0.0, scale=sigma, size=len(series))
    # AR(1): eta[t] = phi_ar * eta[t-1] + eps[t]
    # Equivalent IIR filter H(z) = 1 / (1 - phi_ar·z⁻¹); runs in compiled C
    # via scipy.signal.lfilter — ~100× faster than a Python loop for n=252.
    eta = lfilter([1.0], [1.0, -phi_ar], eps)
    return series + eta


def sample_lppls_params(
    t2: float,
    rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    """Sample random LPPLS nonlinear parameters within spec ranges (spec §3, §7).

    Parameters
    ----------
    t2 : float
        End of observed series (days before normalisation).
    rng : np.random.Generator, optional

    Returns
    -------
    (tc, m, omega) : tuple of float
        tc in [t2, t2 + 50], m in [0.1, 0.9], omega in [6, 13].
    """
    if rng is None:
        rng = np.random.default_rng()
    tc = rng.uniform(t2, t2 + 50.0)
    m = rng.uniform(0.1, 0.9)
    omega = rng.uniform(6.0, 13.0)
    return float(tc), float(m), float(omega)
