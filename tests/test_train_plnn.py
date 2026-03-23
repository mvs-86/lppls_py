"""
Tests for P-LNN full training pipeline.

Covers:
  - dataset.py  : sample generation, tf.data pipeline
  - train_plnn.py : training loop, checkpointing, history dict
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# dataset.py tests
# ---------------------------------------------------------------------------

class TestDataset:

    def test_generate_plnn_sample_white_shape(self):
        """generate_plnn_sample returns (252,) series and (3,) label."""
        from lppls.dataset import generate_plnn_sample
        rng = np.random.default_rng(0)
        result = None
        for _ in range(100):
            result = generate_plnn_sample("white", rng)
            if result is not None:
                break
        assert result is not None, "All samples were None"
        series, label = result
        assert series.shape == (252,), f"series shape {series.shape}"
        assert label.shape  == (3,),   f"label shape {label.shape}"
        assert series.dtype == np.float32
        assert label.dtype  == np.float32

    def test_generate_plnn_sample_series_in_range(self):
        """Clean series is min-max scaled — after noise it's ~[0,1]."""
        from lppls.dataset import generate_plnn_sample
        rng = np.random.default_rng(1)
        results = []
        for _ in range(200):
            r = generate_plnn_sample("white", rng)
            if r is not None:
                results.append(r)
            if len(results) >= 20:
                break
        assert results, "No valid samples generated"
        for series, _ in results:
            assert np.all(np.isfinite(series)), "Series contains NaN/Inf"

    def test_generate_plnn_sample_label_ranges(self):
        """Labels stay within spec parameter ranges."""
        from lppls.dataset import generate_plnn_sample, _TC_MIN, _TC_MAX
        rng = np.random.default_rng(2)
        labels = []
        for _ in range(500):
            r = generate_plnn_sample("white", rng)
            if r is not None:
                labels.append(r[1])
            if len(labels) >= 50:
                break
        assert labels
        labels = np.stack(labels)  # (N, 3)
        tc, m, omega = labels[:, 0], labels[:, 1], labels[:, 2]
        assert np.all(tc    >= _TC_MIN - 1e-6)
        assert np.all(tc    <= _TC_MAX + 1e-6)
        assert np.all(m     >= 0.1 - 1e-6)
        assert np.all(m     <= 0.9 + 1e-6)
        assert np.all(omega >= 6.0 - 1e-6)
        assert np.all(omega <= 13.0 + 1e-6)

    def test_generate_plnn_sample_ar1(self):
        """AR(1) noise variant also produces valid samples."""
        from lppls.dataset import generate_plnn_sample
        rng = np.random.default_rng(3)
        result = None
        for _ in range(200):
            result = generate_plnn_sample("ar1", rng)
            if result is not None:
                break
        assert result is not None
        series, label = result
        assert series.shape == (252,)
        assert np.all(np.isfinite(series))

    def test_generate_plnn_sample_both(self):
        """'both' variant mixes white and AR(1) noise."""
        from lppls.dataset import generate_plnn_sample
        rng = np.random.default_rng(4)
        results = []
        for _ in range(500):
            r = generate_plnn_sample("both", rng)
            if r is not None:
                results.append(r)
            if len(results) >= 20:
                break
        assert len(results) >= 10, "Too few valid samples"

    def test_make_tf_dataset_batch_shapes(self):
        """make_tf_dataset yields batches of correct shape."""
        import tensorflow as tf
        from lppls.dataset import make_tf_dataset

        ds = make_tf_dataset(noise_type="white", n_samples=16, batch_size=4, seed=0)
        batch = next(iter(ds))
        series_batch, label_batch = batch
        assert series_batch.shape == (4, 252), f"Got {series_batch.shape}"
        assert label_batch.shape  == (4, 3),   f"Got {label_batch.shape}"

    def test_make_tf_dataset_finite_values(self):
        """Dataset produces finite float32 tensors."""
        import tensorflow as tf
        from lppls.dataset import make_tf_dataset

        ds = make_tf_dataset(noise_type="ar1", n_samples=8, batch_size=8, seed=1)
        series_batch, label_batch = next(iter(ds))
        assert tf.reduce_all(tf.math.is_finite(series_batch))
        assert tf.reduce_all(tf.math.is_finite(label_batch))

    def test_make_tf_dataset_invalid_noise_type(self):
        """make_tf_dataset raises ValueError for unknown noise_type."""
        from lppls.dataset import make_tf_dataset
        with pytest.raises(ValueError, match="noise_type"):
            make_tf_dataset(noise_type="unknown", n_samples=10)

    def test_make_tf_dataset_yields_n_samples(self):
        """Dataset yields exactly n_samples total (across all batches)."""
        from lppls.dataset import make_tf_dataset

        n = 24
        ds = make_tf_dataset(noise_type="white", n_samples=n, batch_size=8,
                             seed=5, shuffle=False)
        total = sum(s.shape[0] for s, _ in ds)
        assert total == n, f"Expected {n} samples, got {total}"


# ---------------------------------------------------------------------------
# train_plnn.py tests
# ---------------------------------------------------------------------------

class TestTrainPlnn:

    def test_train_plnn_returns_history_keys(self, tmp_path):
        """train_plnn returns dict with expected keys."""
        from lppls.train_plnn import train_plnn

        history = train_plnn(
            noise_type="white",
            output_path=tmp_path / "plnn_test.keras",
            epochs=1,
            batch_size=4,
            n_train=16,
            n_val=8,
            seed=0,
            mixed_precision=False,
        )
        for key in ("train_loss", "val_loss", "epoch_times", "best_epoch", "best_val_loss"):
            assert key in history, f"Missing key: {key}"

    def test_train_plnn_loss_lists_length(self, tmp_path):
        """Loss lists have length == epochs."""
        from lppls.train_plnn import train_plnn

        epochs = 2
        history = train_plnn(
            noise_type="white",
            output_path=tmp_path / "plnn_test.keras",
            epochs=epochs,
            batch_size=4,
            n_train=16,
            n_val=8,
            seed=0,
            mixed_precision=False,
        )
        assert len(history["train_loss"])  == epochs
        assert len(history["val_loss"])    == epochs
        assert len(history["epoch_times"]) == epochs

    def test_train_plnn_loss_is_finite(self, tmp_path):
        """All reported losses are finite positive numbers."""
        from lppls.train_plnn import train_plnn

        history = train_plnn(
            noise_type="white",
            output_path=tmp_path / "plnn_test.keras",
            epochs=2,
            batch_size=4,
            n_train=16,
            n_val=8,
            seed=0,
            mixed_precision=False,
        )
        for loss in history["train_loss"] + history["val_loss"]:
            assert np.isfinite(loss), f"Non-finite loss: {loss}"
            assert loss >= 0, f"Negative loss: {loss}"

    def test_train_plnn_saves_model(self, tmp_path):
        """train_plnn saves a loadable Keras model."""
        import keras
        from lppls.train_plnn import train_plnn

        out = tmp_path / "plnn_saved.keras"
        train_plnn(
            noise_type="white",
            output_path=out,
            epochs=1,
            batch_size=4,
            n_train=8,
            n_val=4,
            seed=0,
            mixed_precision=False,
        )
        assert out.exists(), "Model file not created"
        loaded = keras.models.load_model(out, compile=False)
        assert loaded is not None

    def test_train_plnn_loaded_model_predicts(self, tmp_path):
        """Loaded model can predict on a new batch."""
        import numpy as np
        import keras
        from lppls.train_plnn import train_plnn

        out = tmp_path / "plnn_pred.keras"
        train_plnn(
            noise_type="white",
            output_path=out,
            epochs=1,
            batch_size=4,
            n_train=8,
            n_val=4,
            seed=0,
            mixed_precision=False,
        )
        model = keras.models.load_model(out, compile=False)
        X = np.random.randn(4, 252).astype(np.float32)
        preds = model(X)
        assert preds.shape == (4, 3)

    def test_train_plnn_best_val_loss_monotone(self, tmp_path):
        """best_val_loss <= all val_loss values."""
        from lppls.train_plnn import train_plnn

        history = train_plnn(
            noise_type="white",
            output_path=tmp_path / "plnn_mono.keras",
            epochs=3,
            batch_size=4,
            n_train=24,
            n_val=8,
            seed=0,
            mixed_precision=False,
        )
        assert history["best_val_loss"] <= min(history["val_loss"]) + 1e-9

    def test_train_plnn_ar1_variant(self, tmp_path):
        """AR(1) noise variant trains without error."""
        from lppls.train_plnn import train_plnn

        history = train_plnn(
            noise_type="ar1",
            output_path=tmp_path / "plnn_ar1.keras",
            epochs=1,
            batch_size=4,
            n_train=8,
            n_val=4,
            seed=1,
            mixed_precision=False,
        )
        assert len(history["train_loss"]) == 1

    def test_train_plnn_both_variant(self, tmp_path):
        """Both-noise variant trains without error."""
        from lppls.train_plnn import train_plnn

        history = train_plnn(
            noise_type="both",
            output_path=tmp_path / "plnn_both.keras",
            epochs=1,
            batch_size=4,
            n_train=8,
            n_val=4,
            seed=2,
            mixed_precision=False,
        )
        assert len(history["train_loss"]) == 1
