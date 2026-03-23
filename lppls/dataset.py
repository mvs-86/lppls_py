"""
Synthetic P-LNN dataset generation (spec §7).

Generates (series, label) pairs where:
  - series : np.ndarray, shape (252,), min-max scaled to [0, 1]
  - label  : np.ndarray, shape (3,), values [tc, m, omega] in natural units

Parameter ranges (spec Table 1 / §A.2):
  tc    : [t2, t2 + 50/252]  (normalised time, t2 = 1.0)
  m     : [0.1, 0.9]
  omega : [6.0, 13.0]

Noise types (spec §7.4):
  "white"  → P-LNN-100K       (white noise, alpha in [0.01, 0.15])
  "ar1"    → P-LNN-100K-AR1   (AR(1) noise, sigma in [0.01, 0.05])
  "both"   → P-LNN-100K-BOTH  (50/50 mix)

Dataset modes
-------------
pregenerate=True  (default)
    All samples are generated upfront into two numpy arrays
    (X: float32 (n,252) ≈ 96 MB, Y: float32 (n,3) ≈ 1 MB for n=100 K).
    tf.data.Dataset.from_tensor_slices eliminates the Python-generator
    bottleneck so the GPU pipeline is no longer CPU-starved.

pregenerate=False
    Legacy streaming via tf.data.Dataset.from_generator.
    Useful when RAM is limited.
"""

from __future__ import annotations

from typing import Generator, Literal

import numpy as np
import tensorflow as tf

from .formula import lppls_reformulated
from .data_gen import add_white_noise, add_ar1_noise

# Exported type alias so all modules share a single source of truth.
NoiseType = Literal["white", "ar1", "both"]

# Fixed series length for P-LNN (spec §6.1)
SERIES_LEN = 252

# Normalised time axis: t in [0, 1], t2 = 1.0
_T_AXIS = np.linspace(0.0, 1.0, SERIES_LEN)

# tc range in normalised units: t2=1.0, window = 50 days out of 252
_TC_MIN = 1.0
_TC_MAX = 1.0 + 50.0 / SERIES_LEN  # ≈ 1.198

# Noise amplitude ranges (spec Table 1)
_WHITE_ALPHA_MIN, _WHITE_ALPHA_MAX = 0.01, 0.15
_AR1_SIGMA_MIN,  _AR1_SIGMA_MAX   = 0.01, 0.05

# AR(1) coefficient (spec §7.3)
_PHI_AR = 0.9

# Dataset split sizes (spec §A.2)
N_TRAIN = 100_000
N_VAL   = 33_333


def _sample_nonlinear_params(rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample {tc, m, omega} uniformly within spec ranges."""
    tc    = rng.uniform(_TC_MIN, _TC_MAX)
    m     = rng.uniform(0.1, 0.9)
    omega = rng.uniform(6.0, 13.0)
    return float(tc), float(m), float(omega)


def _sample_linear_params(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Sample {A, B, C1, C2} to produce a valid crash-bubble shape.

    A is fixed; B is negative (power-law growth toward tc); C1, C2 are small
    oscillation amplitudes.  The series is min-max scaled afterward so absolute
    magnitudes do not affect training.
    """
    A  = 1.0
    B  = rng.uniform(-2.0, -0.1)
    C1 = rng.uniform(-0.3, 0.3)
    C2 = rng.uniform(-0.3, 0.3)
    return float(A), float(B), float(C1), float(C2)


def generate_plnn_sample(
    noise_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate one (series, label) pair.

    Parameters
    ----------
    noise_type : {"white", "ar1", "both"}
    rng : np.random.Generator

    Returns
    -------
    (series, label) or None if generation produced NaN/Inf.
      series : shape (252,), float32, values in [0, 1]
      label  : shape (3,),   float32, [tc, m, omega] in natural units
    """
    tc, m, omega = _sample_nonlinear_params(rng)
    A, B, C1, C2 = _sample_linear_params(rng)

    clean = lppls_reformulated(_T_AXIS, tc, m, omega, A, B, C1, C2)

    if not np.all(np.isfinite(clean)):
        return None

    # Min-max scale to [0, 1]
    s_min, s_max = clean.min(), clean.max()
    if s_max - s_min < 1e-12:
        return None
    clean = (clean - s_min) / (s_max - s_min)

    # Add noise
    actual_noise = noise_type
    if noise_type == "both":
        actual_noise = rng.choice(["white", "ar1"])

    if actual_noise == "white":
        alpha = rng.uniform(_WHITE_ALPHA_MIN, _WHITE_ALPHA_MAX)
        noisy = add_white_noise(clean, alpha=alpha, rng=rng)
    else:  # ar1
        sigma = rng.uniform(_AR1_SIGMA_MIN, _AR1_SIGMA_MAX)
        noisy = add_ar1_noise(clean, phi_ar=_PHI_AR, sigma=sigma, rng=rng)

    if not np.all(np.isfinite(noisy)):
        return None

    series = noisy.astype(np.float32)
    label  = np.array([tc, m, omega], dtype=np.float32)
    return series, label


def _sample_stream(
    noise_type: str,
    rng: np.random.Generator,
    n_samples: int,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield exactly *n_samples* valid (series, label) pairs, retrying on None."""
    yielded = 0
    while yielded < n_samples:
        result = generate_plnn_sample(noise_type, rng)
        if result is None:
            continue
        yield result
        yielded += 1


def pregenerate_arrays(
    noise_type: str,
    n_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-generate all samples into contiguous numpy arrays.

    Parameters
    ----------
    noise_type : {"white", "ar1", "both"}
    n_samples : int
    seed : int, optional

    Returns
    -------
    X : np.ndarray, shape (n_samples, 252), float32
    Y : np.ndarray, shape (n_samples, 3),   float32
        Labels [tc, m, omega] in natural units.
    """
    X = np.empty((n_samples, SERIES_LEN), dtype=np.float32)
    Y = np.empty((n_samples, 3),          dtype=np.float32)
    rng = np.random.default_rng(seed)
    for i, (series, label) in enumerate(_sample_stream(noise_type, rng, n_samples)):
        X[i] = series
        Y[i] = label
    return X, Y


def make_tf_dataset(
    noise_type: str,
    n_samples: int,
    batch_size: int = 8,
    seed: int | None = None,
    shuffle: bool = True,
    pregenerate: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (series, label) pairs.

    Parameters
    ----------
    noise_type : {"white", "ar1", "both"}
    n_samples : int
    batch_size : int
        Spec §6.4: 8.
    seed : int, optional
    shuffle : bool
        Shuffle before batching (recommended for training set).
    pregenerate : bool
        If True (default), generate all samples upfront into numpy arrays
        and use from_tensor_slices — eliminates the Python-generator CPU
        bottleneck that starves the GPU.  Uses ~(n * 252 * 4) bytes of RAM
        (≈ 96 MB for n=100 K).
        If False, use the legacy from_generator streaming pipeline.

    Returns
    -------
    tf.data.Dataset
        Yields (series_batch, label_batch) of shape ((batch,252), (batch,3)).
    """
    if noise_type not in ("white", "ar1", "both"):
        raise ValueError(f"noise_type must be 'white', 'ar1', or 'both'; got {noise_type!r}")

    if pregenerate:
        X, Y = pregenerate_arrays(noise_type, n_samples, seed=seed)
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            # reshuffle_each_iteration=True (default) gives a fresh permutation
            # every epoch without needing .repeat() + steps_per_epoch.
            ds = ds.shuffle(buffer_size=n_samples, seed=seed)
    else:
        # ── Legacy streaming path ──────────────────────────────────────────
        def generator():
            rng = np.random.default_rng(seed)
            yield from _sample_stream(noise_type, rng, n_samples)

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(SERIES_LEN,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,),          dtype=tf.float32),
            ),
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=min(n_samples, 10_000), seed=seed)

    # drop_remainder=True: every batch is exactly batch_size samples,
    # preventing uneven per-replica splits in MirroredStrategy.
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
