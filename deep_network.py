from collections import namedtuple
from typing import Callable, Tuple
import numpy as np


Layer = namedtuple("Layer", ["n_neurons", "activation_function"])


class NetworkArchitecture:
    def __init__(self):
        self._architecture: dict = {}
        self._n_layers = 0

    @property
    def architecture(self) -> dict:
        if not self._architecture:
            raise Exception("Network architecture is not defined.")
        return dict(self._architecture)

    @property
    def dimentions(self) -> np.array:
        return np.array([layer.n_neurons for layer in self.architecture.values()])

    def get_layer_neurons_number(self, layer_number: int) -> int:
        return self.architecture[f"layer{layer_number}"]

    def get_weight_matrix_dimention(self, layer_number: int) -> tuple:
        return (self._architecture[f"layer{layer_number}"].n_neurons, self._architecture[f"layer{layer_number - 1}"].n_neurons)

    def add_layer(self, n_neurons: int, activation_function: Callable) -> None:
        """
        Adds layers: layer1, layer2 ... layerN with namedtuple(n_neurons, activation_function)
        - activation_function is Callable and IT MUST BE VECTORIZED!
        """
        self._architecture["layer" + str(self._n_layers)] = Layer(n_neurons, activation_function)
        self._n_layers += 1


class DeepNetwork:
    def __init__(self, network: NetworkArchitecture) -> None:
        self._network_architecture = network
        self._parameters = {}

    def predict(self, X: np.array) -> np.array:
        pass

    def train_model(self, X: np.array, Y: np.array, n_iterations: int, learning_rate: float) -> None:
        if not self._parameters:
            self._parameters = self._generate_parameters(self._network_architecture)
        parameters = self._parameters
        for _ in range(n_iterations):
            A_output, cache = self._propagate_forward(X, parameters)
        

    def _generate_parameters(self, network_architecture: NetworkArchitecture) -> dict:
        parameters = {}
        for layer in range(1, len(network_architecture.dimentions)):
            weight_matrix_shape = network_architecture.get_weight_matrix_dimention(layer)  # (n[l], n[l-1])
            parameters[f"W{layer}"] = np.random.randn(weight_matrix_shape) * np.sqrt(2/weight_matrix_shape[1]) 
            parameters[f"b{layer}"] = np.zeros((weight_matrix_shape[0], 1))
        return parameters

    def _compute_cost(self, A_output: np.array, Y: np.array) -> float:
        pass

    def _propagate_forward(self, X: np.array, parameters: dict) -> Tuple[float, dict]:
        """Returns: tuple of A[L] and cache."""
        pass

    def _propagate_backward(self) -> None:
        """
        dZ[l] = A[l] - Y   for sigmoid activ. func.            || (n[l], m) 
        dW[l] = 1/m * np.dot(dZ[l], A[l-1].T)                  || (n[l], n[l-1])
        db[l] = 1/m * np.sum(dZ[l], axis=1, keepdims=True)     || (n[l], 1)
        dZ[l-1] = np.dot(W2.T, dZ[l]) * d_activation(Z[l-1])   || (n[l-1], m)
        dW[l-1] = 1/m * np.dot(dZ[l-1], A[l-2].T)              || (n[l-1], n[l-2])
        db[l-1] = 1/m * np.sum(dZ[l-1], axis=1, keepdims=True) || (n[l-1], 1)
        """
        pass

    