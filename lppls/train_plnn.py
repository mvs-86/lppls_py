"""
Full P-LNN training pipeline (spec §6.4).

Training procedure
------------------
- Optimizer  : Adam, lr = 1e-5, global_clipnorm=1.0  (spec §6.4)
- Epochs     : 20                                     (spec §6.4)
- Batch size : 8                                      (spec §6.4)
- Dataset    : 100,000 training + 33,333 validation synthetic LPPLS series
               pre-generated into RAM (from_tensor_slices) to saturate GPU
- Loss       : MSE on {tc, m, omega} labels  (spec §6.3)
- GPU        : tf.distribute.MirroredStrategy (multi-GPU transparent)
- Mixed prec : enabled via keras policy "mixed_float16" when GPU available

Usage (CLI)
-----------
    python -m lppls.train_plnn --noise white --output models/plnn_white.keras
    python -m lppls.train_plnn --noise ar1   --output models/plnn_ar1.keras
    python -m lppls.train_plnn --noise both  --output models/plnn_both.keras

Usage (API)
-----------
    from lppls.train_plnn import train_plnn

    history = train_plnn(
        noise_type="white",
        output_path="models/plnn_white.keras",
    )
"""

from __future__ import annotations

import subprocess
import time
import argparse
import logging
from pathlib import Path

import tensorflow as tf
import keras

from .plnn import build_plnn, plnn_loss
from .dataset import make_tf_dataset, N_TRAIN, N_VAL, SERIES_LEN, NoiseType

logger = logging.getLogger(__name__)

# ── Hyperparameters (spec §6.4) ────────────────────────────────────────────
_LR         = 1e-5
_EPOCHS     = 20
_BATCH_SIZE = 8


def _setup_strategy() -> tf.distribute.Strategy:
    """Return MirroredStrategy for multi-GPU or default for CPU/single-GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info("MirroredStrategy: %d GPUs", len(gpus))
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        logger.info("Single GPU: %s", gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()
        logger.info("No GPU found; training on CPU")
    return strategy


def _enable_mixed_precision() -> None:
    """Enable float16 mixed precision when a GPU is available."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled: mixed_float16")
    else:
        logger.info("Mixed precision skipped (no GPU)")


def _query_gpu_util() -> list[dict]:
    """Return per-GPU utilisation stats from nvidia-smi, or [] if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                "memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        rows = []
        for line in result.stdout.strip().splitlines():
            idx, name, gpu_pct, mem_pct, mem_used, mem_total = [
                p.strip() for p in line.split(",")
            ]
            rows.append({
                "index": idx, "name": name,
                "gpu_pct": int(gpu_pct), "mem_pct": int(mem_pct),
                "mem_used_mib": int(mem_used), "mem_total_mib": int(mem_total),
            })
        return rows
    except Exception:
        return []


class _EpochTimer(keras.callbacks.Callback):
    """Records wall-clock time and samples/sec per epoch."""

    def __init__(self, n_train: int) -> None:
        super().__init__()
        self.n_train = n_train
        self.epoch_times: list[float] = []
        self.throughput:  list[float] = []
        self._t0: float = 0.0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        elapsed = time.perf_counter() - self._t0
        sps = self.n_train / elapsed
        self.epoch_times.append(elapsed)
        self.throughput.append(sps)
        logger.info("  Throughput: %.0f samples/sec  (%.1fs/epoch)", sps, elapsed)


def _log_gpu_util(label: str = "") -> None:
    """Log per-GPU utilisation from nvidia-smi; silently skips if unavailable."""
    tag = f" [{label}]" if label else ""
    for gpu in _query_gpu_util():
        logger.info(
            "  GPU %s (%s)%s: compute=%d%%  mem=%d%%  (%d/%d MiB)",
            gpu["index"], gpu["name"], tag,
            gpu["gpu_pct"], gpu["mem_pct"],
            gpu["mem_used_mib"], gpu["mem_total_mib"],
        )


class _GPUUtilLogger(keras.callbacks.Callback):
    """Logs GPU compute & memory utilisation via nvidia-smi after each epoch."""

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        _log_gpu_util()


def train_plnn(
    noise_type: NoiseType = "white",
    output_path: str | Path | None = None,
    epochs: int = _EPOCHS,
    batch_size: int = _BATCH_SIZE,
    lr: float = _LR,
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    seed: int | None = 42,
    mixed_precision: bool = True,
) -> dict:
    """Train a P-LNN model and save to disk.

    Parameters
    ----------
    noise_type : {"white", "ar1", "both"}
        Noise augmentation variant (spec §7.4).
    output_path : str or Path, optional
        Where to save the trained model (.keras format).
        Defaults to ``models/plnn_{noise_type}.keras``.
    epochs : int
        Training epochs (spec: 20).
    batch_size : int
        Batch size (spec: 8).
    lr : float
        Adam learning rate (spec: 1e-5).
    n_train : int
        Training samples (spec: 100,000).
    n_val : int
        Validation samples (spec: 33,333).
    seed : int, optional
        RNG seed for reproducibility.
    mixed_precision : bool
        Enable float16 mixed precision on GPU.

    Returns
    -------
    dict
        Training history with keys:
          "train_loss"   : list[float]  (per epoch)
          "val_loss"     : list[float]  (per epoch)
          "epoch_times"  : list[float]  (seconds per epoch)
          "throughput"   : list[float]  (samples/sec per epoch)
          "best_epoch"   : int
          "best_val_loss": float
    """
    if output_path is None:
        output_path = Path("models") / f"plnn_{noise_type}.keras"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mixed_precision:
        _enable_mixed_precision()

    strategy = _setup_strategy()

    # ── Datasets (pre-generated into RAM) ─────────────────────────────────
    effective_batch = batch_size * strategy.num_replicas_in_sync

    t_data = time.perf_counter()
    logger.info(
        "Pre-generating training dataset (%d samples, noise=%s) …",
        n_train, noise_type,
    )
    train_ds = make_tf_dataset(
        noise_type=noise_type,
        n_samples=n_train,
        batch_size=effective_batch,
        seed=seed,
        shuffle=True,
        pregenerate=True,
    )
    logger.info(
        "Pre-generating validation dataset (%d samples) …", n_val,
    )
    val_ds = make_tf_dataset(
        noise_type=noise_type,
        n_samples=n_val,
        batch_size=effective_batch,
        seed=(seed + 1) if seed is not None else None,
        shuffle=False,
        pregenerate=True,
    )
    logger.info(
        "Datasets ready in %.1fs  (~%.0f MB RAM)",
        time.perf_counter() - t_data,
        (n_train + n_val) * SERIES_LEN * 4 / 1e6,
    )

    # Log baseline GPU state before any training compute
    _log_gpu_util("baseline")

    # ── Model ─────────────────────────────────────────────────────────────
    with strategy.scope():
        model = build_plnn(input_size=SERIES_LEN)
        # global_clipnorm prevents float16 gradient overflow with mixed_float16:
        # the default LossScaleOptimizer starts at scale 2^15, which pushes
        # float16 gradients (max ~65504) to inf → NaN → skipped updates.
        # Clipping before unscaling keeps gradients finite every step.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr, global_clipnorm=1.0),
            loss=plnn_loss,
        )

    logger.info("Model parameters: %s", f"{model.count_params():,}")

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_path = output_path.with_suffix(".ckpt.keras")
    timer = _EpochTimer(n_train=n_train)
    callbacks = [
        timer,
        _GPUUtilLogger(),
        # Halve LR when val_loss stalls for 3 epochs; prevents premature plateau
        # at a fixed lr=1e-5 across all 20 epochs.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    # from_tensor_slices gives a finite dataset — no .repeat() or
    # steps_per_epoch needed; Keras iterates through all batches each epoch.
    keras_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Load best weights, save to final path, clean up checkpoint
    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path, compile=False)
        checkpoint_path.unlink()

    model.save(output_path)
    logger.info("Saved final model → %s", output_path)

    # ── Build history dict ─────────────────────────────────────────────────
    train_losses = keras_history.history["loss"]
    val_losses   = keras_history.history["val_loss"]
    best_epoch   = int(val_losses.index(min(val_losses))) + 1  # 1-based

    return {
        "train_loss":    train_losses,
        "val_loss":      val_losses,
        "epoch_times":   timer.epoch_times,
        "throughput":    timer.throughput,
        "best_epoch":    best_epoch,
        "best_val_loss": min(val_losses),
    }


# ── CLI entry-point ────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a P-LNN model (Deep LPPLS, arXiv:2405.12803v1)"
    )
    p.add_argument(
        "--noise", choices=["white", "ar1", "both"], default="white",
        help="Noise augmentation variant (default: white)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path for the saved model (.keras). "
             "Defaults to models/plnn_{noise}.keras",
    )
    p.add_argument("--epochs",     type=int,   default=_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=_BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=_LR)
    p.add_argument("--n-train",    type=int,   default=N_TRAIN)
    p.add_argument("--n-val",      type=int,   default=N_VAL)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument(
        "--no-mixed-precision", dest="mixed_precision",
        action="store_false", default=True,
        help="Disable float16 mixed precision",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    result = train_plnn(
        noise_type=args.noise,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )

    print(
        f"\nTraining complete."
        f"  Best epoch: {result['best_epoch']}"
        f"  Best val loss: {result['best_val_loss']:.6f}"
        f"  Avg throughput: {sum(result['throughput'])/len(result['throughput']):.0f} samples/sec"
    )
