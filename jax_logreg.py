import jax.numpy as jnp
from jax import grad
import numpy as np

from typing import Tuple


def sigmoid(z: jnp.array):
    return 1 / (1 + jnp.exp(-z))


def init_weights(n_inputs: int) -> Tuple[jnp.array, float]:
    """
    For logreg weights can be initialized with zeros.
    """
    return jnp.zeros(shape=(1, n_inputs) , dtype=float), jnp.zeros((1, 1))


def predict(weights: jnp.array, bias: float, X: jnp.array) -> jnp.array:
    return sigmoid(jnp.dot(weights, X) + bias)


def objective(weights: jnp.array, bias: jnp.array, X: jnp.array, Y: jnp.array) -> float:
    """
    Args:
        weights: shape = (1, nx)
        bias: (1, 1)
        X: shape = (nx, m)
        Y: shape = (1, m)
    Returns:
        Result of binary cross-entropy cost function.
    """
    Y_pred = predict(weights, bias, X)
    return jnp.sum(-(Y * jnp.log(Y_pred) + (1 - Y) * jnp.log(1 - Y_pred)))


def update_parameters(weights: jnp.array, bias: jnp.array, X: jnp.array, Y: jnp.array, learning_rate: float) -> Tuple[jnp.array, jnp.array]:
    weights_grads, bias_grads = grad(objective, argnums=(0, 1))(weights, bias, X, Y)
    updated_weights = weights - learning_rate * weights_grads
    updated_bias = bias - learning_rate * bias_grads
    return updated_weights, updated_bias


if __name__ == "__main__":
    inputs = jnp.transpose(jnp.array([[0.52, 1.12,  0.77],
                        [0.88, -1.08, 0.15],
                        [0.52, 0.06, -1.30],
                        [0.74, -2.49, 1.39]]))
    targets = jnp.array([1, 1, 0, 1])
    weights, bias = init_weights(inputs.shape[0])
    for epoch in range(5000): 
        if epoch % 100 == 0:
            print(objective(weights, bias, inputs, targets))
        weights, bias = update_parameters(weights, bias, inputs, targets, 1e-2)

