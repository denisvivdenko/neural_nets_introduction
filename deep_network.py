from array import array
from collections import namedtuple
from typing import Callable, Tuple
import numpy as np

from dataset.planar_utils import load_planar_dataset


Layer = namedtuple("Layer", ["n_neurons", "activation_function"])


class NetworkArchitecture:
    def __init__(self):
        self._architecture: dict = {}
        self.n_layers = 0

    @property
    def architecture(self) -> dict:
        if not self._architecture:
            raise Exception("Network architecture is not defined.")
        return dict(self._architecture)

    @property
    def dimentions(self) -> np.array:
        return np.array([layer.n_neurons for layer in self.architecture.values()])

    def get_layer_neurons_number(self, layer_number: int) -> int:
        return self.architecture[("layer", layer_number)]

    def get_weight_matrix_dimention(self, layer_number: int) -> tuple:
        return (self._architecture[("layer", layer_number)].n_neurons, self._architecture[("layer", layer_number - 1)].n_neurons)

    def add_layer(self, n_neurons: int, activation_function: Callable = None) -> None:
        """
        Adds layers: layer0, layer1 ... layerN with namedtuple(n_neurons, activation_function)
        layer0 is input layer
        - activation_function is Callable and IT MUST BE VECTORIZED!
        """
        self._architecture[("layer", self._n_layers)] = Layer(n_neurons, activation_function)
        self.n_layers += 1


def sigmoid(z: np.array) -> np.array:
    return 1 / (1 + np.exp(-z))


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
        for layer in range(1, network_architecture.n_layers):
            weight_matrix_shape = network_architecture.get_weight_matrix_dimention(layer)  # (n[l], n[l-1])
            parameters[("W", layer)] = np.random.randn(*weight_matrix_shape) * np.sqrt(2 / weight_matrix_shape[1]) 
            parameters[("b", layer)] = np.zeros((weight_matrix_shape[0], 1))
        return parameters

    def _compute_cost(self, A_output: np.array, Y: np.array) -> float:
        pass

    def _propagate_forward(self, X: np.array, parameters: dict) -> Tuple[float, dict]:
        """Returns: tuple of A[L] and cache."""
        cache = {
            ("A", 0): X
        }
        for layer in range(1, self._network_architecture.n_layers):
            cache[("Z", layer)] = np.dot(parameters[("W", layer)], cache[("A", layer - 1)]) + parameters[("b", layer)]
            cache[("A", layer)] = self._network_architecture.architecture[("layer", layer)].activation_function(cache[("Z", layer)])
        return cache[("A", layer)], cache 

    def _propagate_backward(self, Y: np.array, parameters: dict, cache: dict) -> dict:
        """
        Equations:
            dZ[l] = A[l] - Y   for sigmoid activ. func.            || (n[l], m) 
            dW[l] = 1/m * np.dot(dZ[l], A[l-1].T)                  || (n[l], n[l-1])
            db[l] = 1/m * np.sum(dZ[l], axis=1, keepdims=True)     || (n[l], 1)
            dZ[l-1] = np.dot(W[l].T, dZ[l]) * d_activation(Z[l-1])   || (n[l-1], m)
            dW[l-1] = 1/m * np.dot(dZ[l-1], A[l-2].T)              || (n[l-1], n[l-2])
            db[l-1] = 1/m * np.sum(dZ[l-1], axis=1, keepdims=True) || (n[l-1], 1)
        Returns: (dict) gradients.
        """
        # grads[("dZ", )]
        for layer in range(1, self._network_architecture.n_layers):
            pass

if __name__ == "__main__":
    architecture = NetworkArchitecture()
    architecture.add_layer(2)
    architecture.add_layer(5, np.tanh)
    architecture.add_layer(3, np.tanh)
    architecture.add_layer(1, sigmoid)

    network = DeepNetwork(architecture)

    X, Y = load_planar_dataset()  # (n, m), (1, m)
    network.train_model(X, Y, n_iterations=1, learning_rate=0.01)
