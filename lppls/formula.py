"""
LPPLS formula implementations.

Implements:
  - Eq. 1 (full form): O(t) = A + B(tc-t)^m + C(tc-t)^m * cos(omega*ln(tc-t) - phi)
  - Eq. 4/5 (reformulated): O(t) = A + B*f + C1*g + C2*h
  - Eq. 8 (linear parameter recovery): 4x4 matrix solve
"""

import numpy as np


def lppls(t: np.ndarray, tc: float, m: float, omega: float,
          A: float, B: float, C: float, phi: float) -> np.ndarray:
    """Full LPPLS formula (Eq. 1).

    O(t) = A + B*(tc - t)^m + C*(tc - t)^m * cos(omega * ln(tc - t) - phi)

    Parameters
    ----------
    t : array-like
        Time values (all must be < tc).
    tc : float
        Critical time.
    m : float
        Power-law exponent.
    omega : float
        Log-periodic angular frequency.
    A, B, C : float
        Linear parameters.
    phi : float
        Phase of log-periodic oscillation.

    Returns
    -------
    np.ndarray
        LPPLS values, same shape as t.
    """
    t = np.asarray(t, dtype=float)
    dt = tc - t
    # Guard against non-positive dt (t >= tc would be undefined)
    dt = np.where(dt > 0, dt, np.nan)
    f = dt ** m
    result = A + B * f + C * f * np.cos(omega * np.log(dt) - phi)
    return result


def lppls_reformulated(t: np.ndarray, tc: float, m: float, omega: float,
                        A: float, B: float, C1: float, C2: float) -> np.ndarray:
    """Reformulated LPPLS formula (Eq. 4/5).

    O(t) = A + B*f + C1*g + C2*h

    where:
        f = (tc - t)^m
        g = (tc - t)^m * cos(omega * ln(tc - t))
        h = (tc - t)^m * sin(omega * ln(tc - t))

    C1 = C * cos(phi),  C2 = C * sin(phi)

    Parameters
    ----------
    t : array-like
        Time values (all must be < tc).
    tc : float
        Critical time.
    m : float
        Power-law exponent.
    omega : float
        Log-periodic angular frequency.
    A, B, C1, C2 : float
        Linear parameters.

    Returns
    -------
    np.ndarray
        LPPLS values, same shape as t.
    """
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.where(dt > 0, dt, np.nan)
    log_dt = np.log(dt)
    f = dt ** m
    g = f * np.cos(omega * log_dt)
    h = f * np.sin(omega * log_dt)
    return A + B * f + C1 * g + C2 * h


def _build_basis(t: np.ndarray, tc: float, m: float,
                 omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (f, g, h) basis vectors for given nonlinear params."""
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.where(dt > 0, dt, np.nan)
    log_dt = np.log(dt)
    f = dt ** m
    g = f * np.cos(omega * log_dt)
    h = f * np.sin(omega * log_dt)
    return f, g, h


def recover_linear_params(t: np.ndarray, tc: float, m: float, omega: float,
                           series: np.ndarray) -> tuple[float, float, float, float]:
    """Recover linear parameters {A, B, C1, C2} via Eq. 8 matrix solve.

    At fixed nonlinear params {tc, m, omega}, solves:

        | N      Σf     Σg      Σh     | | A  |   | Σ O_i       |
        | Σf     Σf²    Σfg     Σfh    | | B  | = | Σ f_i O_i   |
        | Σg     Σfg    Σg²     Σgh    | | C1 |   | Σ g_i O_i   |
        | Σh     Σfh    Σgh     Σh²    | | C2 |   | Σ h_i O_i   |

    Parameters
    ----------
    t : array-like
        Normalised time axis.
    tc, m, omega : float
        Fixed nonlinear parameters.
    series : array-like
        Observed log-price series O(t_i).

    Returns
    -------
    (A, B, C1, C2) : tuple of float
    """
    t = np.asarray(t, dtype=float)
    O = np.asarray(series, dtype=float)
    n = len(t)

    f, g, h = _build_basis(t, tc, m, omega)

    # Build 4x4 matrix
    mat = np.array([
        [n,         f.sum(),    g.sum(),    h.sum()   ],
        [f.sum(),   (f*f).sum(), (f*g).sum(), (f*h).sum()],
        [g.sum(),   (f*g).sum(), (g*g).sum(), (g*h).sum()],
        [h.sum(),   (f*h).sum(), (g*h).sum(), (h*h).sum()],
    ])

    rhs = np.array([
        O.sum(),
        (f * O).sum(),
        (g * O).sum(),
        (h * O).sum(),
    ])

    params = np.linalg.solve(mat, rhs)
    A, B, C1, C2 = params
    return float(A), float(B), float(C1), float(C2)
