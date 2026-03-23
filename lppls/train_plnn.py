"""
Full P-LNN training pipeline (spec §6.4).

Training procedure
------------------
- Optimizer  : Adam, lr = 1e-5  (spec §6.4)
- Epochs     : 20               (spec §6.4)
- Batch size : 8                (spec §6.4)
- Dataset    : 100,000 training + 33,333 validation synthetic LPPLS series
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

import time
import argparse
import logging
from pathlib import Path
from typing import Literal

import tensorflow as tf
import keras

from .plnn import build_plnn, plnn_loss
from .dataset import make_tf_dataset, N_TRAIN, N_VAL

logger = logging.getLogger(__name__)

NoiseType = Literal["white", "ar1", "both"]

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


class _EpochTimer(keras.callbacks.Callback):
    """Records wall-clock time per epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_times: list[float] = []
        self._t0: float = 0.0

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.epoch_times.append(time.perf_counter() - self._t0)


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

    # ── Datasets ──────────────────────────────────────────────────────────
    logger.info("Building training dataset (%d samples, noise=%s) …", n_train, noise_type)
    train_ds = make_tf_dataset(
        noise_type=noise_type,
        n_samples=n_train,
        batch_size=batch_size * strategy.num_replicas_in_sync,
        seed=seed,
        shuffle=True,
    )

    logger.info("Building validation dataset (%d samples) …", n_val)
    val_ds = make_tf_dataset(
        noise_type=noise_type,
        n_samples=n_val,
        batch_size=batch_size * strategy.num_replicas_in_sync,
        seed=(seed + 1) if seed is not None else None,
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    with strategy.scope():
        model = build_plnn(input_size=252)
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
    timer = _EpochTimer()
    callbacks = [
        timer,
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    # .repeat() makes the generators infinite; steps_per_epoch/validation_steps
    # tell Keras where each epoch boundary is.
    effective_batch = batch_size * strategy.num_replicas_in_sync
    steps_per_epoch  = n_train // effective_batch
    validation_steps = n_val   // effective_batch

    keras_history = model.fit(
        train_ds.repeat(),
        validation_data=val_ds.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
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
    )
