"""
Tests for Deep LPPLS implementation.
Written BEFORE implementation (TDD).
Covers all spec items in model_spec.md.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_time_axis(n=100, t1=0.0, t2=1.0):
    """Return a normalised time axis of length n in [t1, t2]."""
    return np.linspace(t1, t2, n)


# ---------------------------------------------------------------------------
# Formula tests  (lppls/formula.py)
# ---------------------------------------------------------------------------

class TestLpplsFormula:
    """Eq. 1: O(t) = A + B(tc-t)^m + C(tc-t)^m * cos(omega*ln(tc-t) - phi)"""

    def test_lppls_formula_shape(self):
        """lppls() returns array of same shape as t."""
        from lppls.formula import lppls

        t = make_time_axis(n=100)
        tc, m, omega = 1.2, 0.5, 7.0
        A, B, C, phi = 1.0, -0.5, 0.1, 0.3

        result = lppls(t, tc, m, omega, A, B, C, phi)

        assert result.shape == t.shape, (
            f"Expected shape {t.shape}, got {result.shape}"
        )

    def test_lppls_formula_at_known_values(self):
        """O(t) = A when B=C=0 for any t < tc (spec §1.1)."""
        from lppls.formula import lppls

        t = make_time_axis(n=50)
        tc = 1.5
        A = 2.0

        result = lppls(t, tc, m=0.5, omega=7.0, A=A, B=0.0, C=0.0, phi=0.0)

        np.testing.assert_allclose(
            result, A, atol=1e-10,
            err_msg="When B=C=0, O(t) must equal A everywhere"
        )

    def test_lppls_reformulated_equivalence(self):
        """Eq.1 and Eq.4/5 give identical results (spec §1.2).

        C1 = C*cos(phi), C2 = C*sin(phi).
        """
        from lppls.formula import lppls, lppls_reformulated

        t = make_time_axis(n=80)
        tc, m, omega = 1.3, 0.4, 8.0
        A, B, C, phi = 0.5, -0.3, 0.2, 1.1

        C1 = C * np.cos(phi)
        C2 = C * np.sin(phi)

        result_full = lppls(t, tc, m, omega, A, B, C, phi)
        result_reform = lppls_reformulated(t, tc, m, omega, A, B, C1, C2)

        np.testing.assert_allclose(
            result_full, result_reform, atol=1e-10,
            err_msg="Full form and reformulated form must agree"
        )

    def test_linear_params_recovery(self):
        """Eq.8: given {tc,m,omega,A,B,C1,C2}, generate clean series then
        recover {A,B,C1,C2} via matrix solve; recovered params ≈ true to 1e-6.
        """
        from lppls.formula import lppls_reformulated, recover_linear_params

        t = make_time_axis(n=200)
        tc, m, omega = 1.3, 0.45, 7.5
        A, B, C1, C2 = 1.2, -0.8, 0.15, -0.12

        series = lppls_reformulated(t, tc, m, omega, A, B, C1, C2)

        A_hat, B_hat, C1_hat, C2_hat = recover_linear_params(t, tc, m, omega, series)

        np.testing.assert_allclose(A_hat, A, atol=1e-6, err_msg="A recovery failed")
        np.testing.assert_allclose(B_hat, B, atol=1e-6, err_msg="B recovery failed")
        np.testing.assert_allclose(C1_hat, C1, atol=1e-6, err_msg="C1 recovery failed")
        np.testing.assert_allclose(C2_hat, C2, atol=1e-6, err_msg="C2 recovery failed")

    def test_lppls_reconstruction_from_recovered_params(self):
        """After recovering linear params via Eq.8, reconstructed series
        matches original to 1e-6 (spec §8).
        """
        from lppls.formula import lppls_reformulated, recover_linear_params

        t = make_time_axis(n=150)
        tc, m, omega = 1.4, 0.3, 9.0
        A, B, C1, C2 = 0.7, -0.5, 0.08, 0.11

        original = lppls_reformulated(t, tc, m, omega, A, B, C1, C2)

        A_hat, B_hat, C1_hat, C2_hat = recover_linear_params(t, tc, m, omega, original)
        reconstructed = lppls_reformulated(t, tc, m, omega, A_hat, B_hat, C1_hat, C2_hat)

        np.testing.assert_allclose(
            reconstructed, original, atol=1e-6,
            err_msg="Reconstructed series must match original after linear param recovery"
        )


# ---------------------------------------------------------------------------
# Data generation tests  (lppls/data_gen.py)
# ---------------------------------------------------------------------------

class TestDataGen:
    """Spec §7: synthetic data generation."""

    def test_generate_clean_series_shape(self):
        """generate_lppls_series returns array of length n (spec §7.1)."""
        from lppls.data_gen import generate_lppls_series

        n = 252
        series = generate_lppls_series(
            tc=1.1, m=0.5, omega=7.0,
            A=1.0, B=-0.5, C1=0.1, C2=-0.1,
            n=n, t2=1.0
        )

        assert series.shape == (n,), f"Expected ({n},), got {series.shape}"

    def test_generate_clean_series_rescaled(self):
        """Clean series is rescaled to [0, 1] (spec §7.1)."""
        from lppls.data_gen import generate_lppls_series

        series = generate_lppls_series(
            tc=1.1, m=0.5, omega=7.0,
            A=1.0, B=-0.5, C1=0.1, C2=-0.1,
            n=252, t2=1.0
        )

        assert series.min() >= -1e-9, f"Min {series.min()} below 0"
        assert series.max() <= 1.0 + 1e-9, f"Max {series.max()} above 1"
        # At least one value should be ≈ 0 and one ≈ 1 after min-max scaling
        assert abs(series.min()) < 1e-9, "Min of rescaled series should be 0"
        assert abs(series.max() - 1.0) < 1e-9, "Max of rescaled series should be 1"

    def test_generate_white_noise_series(self):
        """Noisy series has same length; noise std matches alpha within tolerance
        (spec §7.2).
        """
        from lppls.data_gen import generate_lppls_series, add_white_noise

        rng = np.random.default_rng(42)
        n = 10_000  # large n for reliable std estimate
        clean = generate_lppls_series(
            tc=1.1, m=0.5, omega=7.0,
            A=1.0, B=-0.5, C1=0.1, C2=-0.1,
            n=n, t2=1.0
        )

        alpha = 0.05
        noisy = add_white_noise(clean, alpha=alpha, rng=rng)

        assert noisy.shape == clean.shape, "Shape mismatch after white noise"

        noise = noisy - clean
        measured_std = noise.std()
        # Allow ±20% relative tolerance for randomness
        assert abs(measured_std - alpha) < 0.2 * alpha, (
            f"Expected std ≈ {alpha}, got {measured_std}"
        )

    def test_generate_ar1_noise_series(self):
        """AR(1) series: variance matches sigma^2/(1-phi_ar^2) within tolerance
        (spec §7.3).
        """
        from lppls.data_gen import generate_lppls_series, add_ar1_noise

        rng = np.random.default_rng(0)
        n = 50_000
        clean = generate_lppls_series(
            tc=1.1, m=0.5, omega=7.0,
            A=1.0, B=-0.5, C1=0.1, C2=-0.1,
            n=n, t2=1.0
        )

        phi_ar = 0.9
        sigma = 0.02   # innovation std
        noisy = add_ar1_noise(clean, phi_ar=phi_ar, sigma=sigma, rng=rng)

        assert noisy.shape == clean.shape, "Shape mismatch after AR(1) noise"

        noise = noisy - clean
        expected_var = sigma**2 / (1 - phi_ar**2)
        measured_var = noise.var()
        # Allow ±30% relative tolerance
        assert abs(measured_var - expected_var) < 0.3 * expected_var, (
            f"Expected var ≈ {expected_var:.6f}, got {measured_var:.6f}"
        )

    def test_parameter_ranges(self):
        """Generated params stay within spec ranges (spec §3, §7).

        tc in [t2, t2+50] (before normalisation),
        m in [0.1, 0.9], omega in [6, 13].
        """
        from lppls.data_gen import sample_lppls_params

        rng = np.random.default_rng(7)
        t2_days = 252.0  # representative trading-day end

        for _ in range(1000):
            tc, m, omega = sample_lppls_params(t2=t2_days, rng=rng)

            assert t2_days <= tc <= t2_days + 50, (
                f"tc={tc} outside [{t2_days}, {t2_days+50}]"
            )
            assert 0.1 <= m <= 0.9, f"m={m} outside [0.1, 0.9]"
            assert 6.0 <= omega <= 13.0, f"omega={omega} outside [6, 13]"


# ---------------------------------------------------------------------------
# M-LNN tests  (lppls/mlnn.py)
# ---------------------------------------------------------------------------

class TestMLNN:
    """Spec §5: M-LNN (Mono-LPPLS-NN)."""

    def test_mlnn_build_output_shape(self):
        """model(X) output shape is (batch, 3) for X of shape (batch, n)
        (spec §5.2).
        """
        import numpy as np
        from lppls.mlnn import build_mlnn

        n = 100
        model = build_mlnn(n=n)

        X = np.random.randn(4, n).astype(np.float32)
        Y = model(X)

        assert Y.shape == (4, 3), f"Expected (4, 3), got {Y.shape}"

    def test_mlnn_penalty_loss_zero_inside_bounds(self):
        """Penalty is 0 when params are within spec bounds (spec §5.3)."""
        import tensorflow as tf
        from lppls.mlnn import penalty_loss

        # tc normalised to [0,1]; m in [0.1, 0.9]; omega in [6, 13]
        params = tf.constant([[0.9, 0.5, 9.0]], dtype=tf.float32)
        loss = penalty_loss(params, alpha=1.0)

        np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-7,
                                   err_msg="Penalty must be 0 inside bounds")

    def test_mlnn_penalty_loss_positive_outside_bounds(self):
        """Penalty > 0 when a param violates bounds (spec §5.3)."""
        import tensorflow as tf
        from lppls.mlnn import penalty_loss

        # omega > 13 violates upper bound
        params = tf.constant([[0.9, 0.5, 15.0]], dtype=tf.float32)
        loss = penalty_loss(params, alpha=1.0)

        assert loss.numpy() > 0, "Penalty must be > 0 when bounds are violated"

    def test_mlnn_total_loss_components(self):
        """L_total = L_mse + L_penalty (spec §5.3)."""
        import tensorflow as tf
        from lppls.mlnn import compute_total_loss

        # Build a minimal scenario: series of length n
        n = 50
        # Random input series
        X = tf.constant(np.random.rand(1, n).astype(np.float32))
        # Params within bounds
        params = tf.constant([[0.9, 0.5, 9.0]], dtype=tf.float32)

        total, mse, penalty = compute_total_loss(X, params, alpha=1.0)

        np.testing.assert_allclose(
            total.numpy(), mse.numpy() + penalty.numpy(), atol=1e-6,
            err_msg="L_total must equal L_mse + L_penalty"
        )

    def test_mlnn_training_reduces_loss(self):
        """After 5 epochs on synthetic data, loss decreases (spec §5.4)."""
        import tensorflow as tf
        from lppls.mlnn import build_mlnn, train_mlnn_one_series
        from lppls.data_gen import generate_lppls_series

        series = generate_lppls_series(
            tc=1.1, m=0.5, omega=7.0,
            A=1.0, B=-0.5, C1=0.1, C2=-0.1,
            n=100, t2=1.0
        )

        model, history = train_mlnn_one_series(series, epochs=5, lr=1e-2, alpha=1.0)

        losses = history["loss"]
        assert len(losses) == 5, "Should record 5 epoch losses"
        assert losses[-1] <= losses[0], (
            f"Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# P-LNN tests  (lppls/plnn.py)
# ---------------------------------------------------------------------------

class TestPLNN:
    """Spec §6: P-LNN (Poly-LPPLS-NN)."""

    def test_plnn_build_output_shape(self):
        """model(X) output shape is (batch, 3) for X shape (batch, 252)
        (spec §6.2).
        """
        import numpy as np
        from lppls.plnn import build_plnn

        model = build_plnn()

        X = np.random.randn(8, 252).astype(np.float32)
        Y = model(X)

        assert Y.shape == (8, 3), f"Expected (8, 3), got {Y.shape}"

    def test_plnn_four_hidden_layers(self):
        """P-LNN has exactly 4 hidden ReLU layers and 1 linear output
        (spec §6.2 Eq.3).
        """
        from lppls.plnn import build_plnn
        import keras

        model = build_plnn()

        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]

        def _activation_name(layer):
            act = layer.activation
            # Keras 3 / TF2 activation can be a function or object
            if hasattr(act, "__name__"):
                return act.__name__
            if hasattr(act, "name"):
                return act.name
            return str(act).lower()

        relu_layers = [l for l in dense_layers if "relu" in _activation_name(l)]
        linear_layers = [
            l for l in dense_layers
            if _activation_name(l) in ("linear", "identity")
        ]

        assert len(relu_layers) == 4, (
            f"Expected 4 ReLU hidden layers, found {len(relu_layers)}: "
            f"{[l.name for l in relu_layers]}"
        )
        assert len(linear_layers) == 1, (
            f"Expected 1 linear output layer, found {len(linear_layers)}"
        )

    def test_plnn_training_step(self):
        """Single training step on a batch of 8 synthetic series runs without
        error (spec §6.4).
        """
        import tensorflow as tf
        from lppls.plnn import build_plnn
        from lppls.data_gen import generate_lppls_series, sample_lppls_params

        rng = np.random.default_rng(1)
        batch_size = 8
        n = 252

        X_batch = []
        Y_batch = []

        for _ in range(batch_size):
            tc, m, omega = sample_lppls_params(t2=n, rng=rng)
            # Random linear params
            A = rng.uniform(0.5, 2.0)
            B = rng.uniform(-1.0, -0.1)
            C1 = rng.uniform(-0.2, 0.2)
            C2 = rng.uniform(-0.2, 0.2)
            s = generate_lppls_series(tc=tc / n, m=m, omega=omega,
                                      A=A, B=B, C1=C1, C2=C2,
                                      n=n, t2=1.0)
            X_batch.append(s)
            # Normalise tc to [0,1] for labels
            Y_batch.append([tc / n, m, omega])

        X = tf.constant(np.array(X_batch, dtype=np.float32))
        Y = tf.constant(np.array(Y_batch, dtype=np.float32))

        model = build_plnn()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

        with tf.GradientTape() as tape:
            preds = model(X, training=True)
            loss = tf.reduce_mean(tf.square(Y - preds))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        assert loss.numpy() >= 0, "Loss must be non-negative"

    def test_plnn_param_mse_loss(self):
        """P-LNN loss is MSE on {tc, m, omega} labels, not on reconstructed
        series (spec §6.3).
        """
        import tensorflow as tf
        from lppls.plnn import plnn_loss

        y_true = tf.constant([[1.05, 0.5, 8.0]], dtype=tf.float32)
        y_pred = tf.constant([[1.05, 0.5, 8.0]], dtype=tf.float32)

        loss_perfect = plnn_loss(y_true, y_pred)
        np.testing.assert_allclose(loss_perfect.numpy(), 0.0, atol=1e-7,
                                   err_msg="Perfect predictions should give 0 loss")

        y_pred_off = tf.constant([[1.10, 0.6, 9.0]], dtype=tf.float32)
        loss_off = plnn_loss(y_true, y_pred_off)
        assert loss_off.numpy() > 0, "Imperfect predictions should give positive loss"
