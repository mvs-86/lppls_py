"""
P-LNN (Poly-LPPLS-NN) implementation (spec §6).

Architecture (Eq. 3):
    h1 = ReLU(W1*X + b1)     [252 → 252]
    h2 = ReLU(W2*h1 + b2)    [252 → 252]
    h3 = ReLU(W3*h2 + b3)    [252 → 252]
    h4 = ReLU(W4*h3 + b4)    [252 → 252]
    Y  = W5*h4 + b5            [252 → 3]

Output Y = [tc, m, omega].

Loss (spec §6.3):
    L = MSE(Y_true, Y_pred) = (1/3) Σ (theta_k_true - theta_k_pred)^2
"""

import tensorflow as tf
import keras
from keras import layers

_INPUT_SIZE = 252


def build_plnn(input_size: int = _INPUT_SIZE) -> keras.Model:
    """Build the P-LNN feed-forward model (spec §6.2).

    Parameters
    ----------
    input_size : int
        Fixed input length. Spec mandates 252.

    Returns
    -------
    keras.Model
        Uncompiled model with output shape (batch, 3).
    """
    strategy = tf.distribute.get_strategy()

    with strategy.scope():
        inputs = keras.Input(shape=(input_size,), name="series_input")
        h1 = layers.Dense(input_size, activation="relu", name="hidden_1")(inputs)
        h2 = layers.Dense(input_size, activation="relu", name="hidden_2")(h1)
        h3 = layers.Dense(input_size, activation="relu", name="hidden_3")(h2)
        h4 = layers.Dense(input_size, activation="relu", name="hidden_4")(h3)
        outputs = layers.Dense(3, activation="linear", name="output")(h4)
        model = keras.Model(inputs=inputs, outputs=outputs, name="P_LNN")

    return model


def plnn_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """P-LNN loss: MSE on {tc, m, omega} (spec §6.3).

    L = MSE(y_true, y_pred) = mean over batch of (1/3) Σ_k (y_true_k - y_pred_k)^2

    Parameters
    ----------
    y_true : tf.Tensor, shape (batch, 3)
    y_pred : tf.Tensor, shape (batch, 3)

    Returns
    -------
    tf.Tensor, scalar
    """
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred), axis=-1))
