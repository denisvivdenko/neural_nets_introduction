from collections import namedtuple
from typing import Callable, Dict, Tuple
import numpy as np

from dataset.planar_utils import load_planar_dataset


Layer = namedtuple("Layer", ["n_neurons", "activation_function"])


class NetworkArchitecture:
    def __init__(self):
        self._architecture: Dict[Layer] = {}
        self.n_layers = 0

    def __getitem__(self, key) -> Layer:
        if not self._architecture:
            raise Exception("Network architecture is not defined.")
        return self._architecture[key]

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
        self._architecture[("layer", self.n_layers)] = Layer(n_neurons, activation_function)
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
        parameters = dict(self._parameters)
        for _ in range(n_iterations):
            A_output, cache = self._propagate_forward(X, parameters)
            print(cache.keys())
            grads = self._propagate_backward(X, Y, parameters, cache)
            parameters = self._update_parameters(parameters, grads, learning_rate)
        
    def _generate_parameters(self, network_architecture: NetworkArchitecture) -> dict:
        parameters = {}
        for layer in range(1, network_architecture.n_layers):
            weight_matrix_shape = network_architecture.get_weight_matrix_dimention(layer)  # (n[l], n[l-1])
            parameters[("W", layer)] = np.random.randn(*weight_matrix_shape) * np.sqrt(2 / weight_matrix_shape[1]) 
            parameters[("b", layer)] = np.zeros((weight_matrix_shape[0], 1))
        return parameters

    def _compute_cost(self, A_output: np.array, Y: np.array) -> float:
        pass

    def _update_parameters(self, parameters: dict, grads: dict, learning_rate: float) -> dict:
        parameters = dict(parameters)
        for key, parameter_value in parameters.items():
            parameter_name, layer = key
            print(parameter_value.shape)
            parameters[(parameter_name, layer)] = parameter_value - learning_rate * grads[(f"d{parameter_name}", layer)]
        return parameters

    def _propagate_forward(self, X: np.array, parameters: dict) -> Tuple[float, dict]:
        """Returns: tuple of A[L] and cache."""
        cache = {
            ("A", 0): X
        }
        for layer in range(1, self._network_architecture.n_layers):
            cache[("Z", layer)] = np.dot(parameters[("W", layer)], cache[("A", layer - 1)]) + parameters[("b", layer)]
            cache[("A", layer)] = self._network_architecture[("layer", layer)].activation_function(cache[("Z", layer)])
        return cache[("A", layer)], cache 

    def _propagate_backward(self, X: np.array, Y: np.array, parameters: dict, cache: dict) -> dict:
        """
        Equations:
            dZ[l] = A[l] - Y   for sigmoid activ. func.            || (n[l], m) 
            dW[l] = 1/m * np.dot(dZ[l], A[l-1].T)                  || (n[l], n[l-1])
            db[l] = 1/m * np.sum(dZ[l], axis=1, keepdims=True)     || (n[l], 1)
            dZ[l-1] = np.dot(W[l].T, dZ[l]) * d_activation(Z[l-1])   || (n[l-1], m)
            dW[l-1] = 1/m * np.dot(dZ[l-1], A[l-1].T)              || (n[l-1], n[l-2])
            db[l-1] = 1/m * np.sum(dZ[l-1], axis=1, keepdims=True) || (n[l-1], 1)
        Returns: (dict) gradients.
        """
        m = X.shape[1]
        grads = {
            ("dZ", 0): X
        }
        output_layer = self._network_architecture.n_layers - 1
        output_layer_activation_function = self._network_architecture[("layer", output_layer)].activation_function
        if output_layer_activation_function.__name__ == sigmoid.__name__:
            grads[("dZ", output_layer)] = cache[("A", output_layer)] - Y
            grads[("dW", output_layer)] = np.dot(grads[("dZ", output_layer)], cache[("A", output_layer - 1)].T) / m
            grads[("db", output_layer)] = np.sum(grads[("dZ", output_layer)], axis=1, keepdims=True) / m

        for layer in range(output_layer - 1, 0, -1):
            d_activation = None
            if self._network_architecture[("layer", layer)].activation_function == np.tanh:
                d_activation  = np.where(cache["Z", layer] > 0, 1, 0)
            else:
                raise Exception("Not implemented.")
    
            grads[("dZ", layer)] = np.dot(parameters[("W", layer + 1)].T, grads[("dZ", layer + 1)]) * d_activation
            grads[("dW", layer)] = np.dot(grads["dZ", layer], cache[("A", layer)].T) / m
            grads[("db", layer)] = np.sum(grads[("dZ", layer)], axis=1, keepdims=True) / m
        return grads


if __name__ == "__main__":
    architecture = NetworkArchitecture()
    architecture.add_layer(2)
    architecture.add_layer(5, np.tanh)
    architecture.add_layer(3, np.tanh)
    architecture.add_layer(1, sigmoid)

    network = DeepNetwork(architecture)

    X, Y = load_planar_dataset()  # (n, m), (1, m)
    network.train_model(X, Y, n_iterations=1, learning_rate=0.01)
