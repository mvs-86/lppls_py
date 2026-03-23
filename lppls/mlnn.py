"""
M-LNN (Mono-LPPLS-NN) implementation (spec §5).

Architecture (Eq. 2):
    h1 = ReLU(W1*X + b1)     [n → n]
    h2 = ReLU(W2*h1 + b2)    [n → n]
    Y  = W_o*h2 + b_o         [n → 3]

Output Y = [tc, m, omega] (nonlinear LPPLS parameters).

Loss:
    L_MSE     = MSE(X, LPPLS(Y))          reconstruction loss
    L_Penalty = alpha * sum_k [max(0, theta_min_k - theta_k) +
                                max(0, theta_k - theta_max_k)]
    L_Total   = L_MSE + L_Penalty
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from .formula import lppls_reformulated, recover_linear_params


# Parameter bounds (spec §3, normalised tc in [0, 1])
_BOUNDS = {
    "tc":    (0.8, 1.2),   # t2 ± 20%; on [0,1] axis t2=1, so [0.8, 1.2]
    "m":     (0.1, 1.0),
    "omega": (6.0, 13.0),
}
_THETA_MIN = tf.constant(
    [_BOUNDS["tc"][0], _BOUNDS["m"][0], _BOUNDS["omega"][0]], dtype=tf.float32
)
_THETA_MAX = tf.constant(
    [_BOUNDS["tc"][1], _BOUNDS["m"][1], _BOUNDS["omega"][1]], dtype=tf.float32
)


def build_mlnn(n: int) -> keras.Model:
    """Build the M-LNN feed-forward model (spec §5.2).

    Parameters
    ----------
    n : int
        Length of the input time series (controls hidden layer width).

    Returns
    -------
    keras.Model
        Uncompiled model with output shape (batch, 3).
    """
    strategy = tf.distribute.get_strategy()

    with strategy.scope():
        inputs = keras.Input(shape=(n,), name="series_input")
        h1 = layers.Dense(n, activation="relu", name="hidden_1")(inputs)
        h2 = layers.Dense(n, activation="relu", name="hidden_2")(h1)
        outputs = layers.Dense(3, activation="linear", name="output")(h2)
        model = keras.Model(inputs=inputs, outputs=outputs, name="M_LNN")

    return model


def penalty_loss(params: tf.Tensor, alpha: float = 1.0) -> tf.Tensor:
    """Compute parameter bound penalty (spec §5.3).

    L_Penalty = alpha * Σ_k [max(0, theta_min_k - theta_k) +
                              max(0, theta_k - theta_max_k)]

    Parameters
    ----------
    params : tf.Tensor, shape (batch, 3)
        Estimated [tc, m, omega] per sample.
    alpha : float
        Penalty coefficient.

    Returns
    -------
    tf.Tensor, scalar
        Mean penalty over batch.
    """
    lower_violation = tf.maximum(0.0, _THETA_MIN - params)
    upper_violation = tf.maximum(0.0, params - _THETA_MAX)
    penalty = alpha * tf.reduce_sum(lower_violation + upper_violation, axis=-1)
    return tf.reduce_mean(penalty)


def _reconstruct_series_differentiable(
    X: tf.Tensor, params: tf.Tensor, n: int
) -> tf.Tensor:
    """Reconstruct LPPLS series via fully TF-differentiable ops (Eq. 4/5/8).

    Computes basis functions f, g, h in TF, then solves the 4×4 normal
    equations (Eq. 8) with tf.linalg.solve to obtain {A, B, C1, C2}, then
    returns the reconstructed series.  All ops are differentiable so gradients
    flow back through params to the model weights.

    Parameters
    ----------
    X : tf.Tensor, shape (batch, n)
        Observed (min-max scaled) time series.
    params : tf.Tensor, shape (batch, 3)
        Estimated [tc, m, omega].
    n : int

    Returns
    -------
    tf.Tensor, shape (batch, n)
    """
    t = tf.cast(tf.linspace(0.0, 1.0, n), tf.float32)  # (n,)
    t = tf.expand_dims(t, 0)  # (1, n)

    tc = params[:, 0:1]   # (batch, 1)
    m  = params[:, 1:2]   # (batch, 1)
    om = params[:, 2:3]   # (batch, 1)

    tau = tc - t                                  # (batch, n)
    tau_safe = tf.maximum(tau, 1e-8)              # avoid log(0) / 0^m

    log_tau = tf.math.log(tau_safe)               # (batch, n)
    f = tf.exp(m * log_tau)                       # (batch, n) = tau^m
    g = f * tf.cos(om * log_tau)                  # (batch, n)
    h = f * tf.sin(om * log_tau)                  # (batch, n)
    ones = tf.ones_like(f)                        # (batch, n)

    # Design matrix D: (batch, n, 4)  columns = [1, f, g, h]
    D = tf.stack([ones, f, g, h], axis=-1)        # (batch, n, 4)

    # Normal equations: (D^T D) beta = D^T X
    Dt = tf.transpose(D, perm=[0, 2, 1])          # (batch, 4, n)
    DtD = tf.matmul(Dt, D)                        # (batch, 4, 4)

    # Add small ridge to ensure invertibility
    ridge = 1e-8 * tf.eye(4, batch_shape=[tf.shape(params)[0]])
    DtD = DtD + ridge

    X_col = tf.expand_dims(tf.cast(X, tf.float32), -1)  # (batch, n, 1)
    DtX = tf.matmul(Dt, X_col)                           # (batch, 4, 1)

    beta = tf.linalg.solve(DtD, DtX)             # (batch, 4, 1)

    reconstructed = tf.matmul(D, beta)            # (batch, n, 1)
    return tf.squeeze(reconstructed, axis=-1)     # (batch, n)


def compute_total_loss(
    X: tf.Tensor,
    params: tf.Tensor,
    alpha: float = 1.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute L_Total = L_MSE + L_Penalty (spec §5.3).

    Parameters
    ----------
    X : tf.Tensor, shape (batch, n)
        Input series (observed log-prices, min-max scaled).
    params : tf.Tensor, shape (batch, 3)
        Network output [tc, m, omega].
    alpha : float
        Penalty coefficient.

    Returns
    -------
    (L_total, L_mse, L_penalty) : tuple of scalar tensors
    """
    n = X.shape[-1] or tf.shape(X)[-1]

    reconstructed = _reconstruct_series_differentiable(X, params, int(n))
    X_float = tf.cast(X, tf.float32)

    l_mse = tf.reduce_mean(tf.square(X_float - reconstructed))
    l_penalty = penalty_loss(params, alpha=alpha)
    l_total = l_mse + l_penalty

    return l_total, l_mse, l_penalty


def train_mlnn_one_series(
    series: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-2,
    alpha: float = 1.0,
) -> tuple[keras.Model, dict]:
    """Train a new M-LNN on a single time series (spec §5.4).

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        Min-max scaled input series.
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate (spec: 1e-2).
    alpha : float
        Penalty coefficient.

    Returns
    -------
    (model, history) : tuple
        Trained model and dict with key "loss" (list of per-epoch losses).
    """
    n = len(series)
    model = build_mlnn(n=n)
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Input tensor: shape (1, n)
    X = tf.constant(series[np.newaxis, :], dtype=tf.float32)

    history = {"loss": []}
    best_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            params = model(X, training=True)
            total_loss, _, _ = compute_total_loss(X, params, alpha=alpha)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_val = float(total_loss.numpy())
        history["loss"].append(loss_val)

        # Checkpoint: save best weights
        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = [w.numpy().copy() for w in model.weights]

    # Restore best weights
    if best_weights is not None:
        for w, w_val in zip(model.weights, best_weights):
            w.assign(w_val)

    return model, history
