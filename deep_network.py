from collections import namedtuple
from typing import Callable
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

    def add_layer(self, n_neurons: int, activation_function: Callable) -> None:
        """
        Adds layers: layer1, layer2 ... layerN with namedtuple(n_neurons, activation_function)
        - activation_function is Callable and IT MUST BE VECTORIZED!
        """
        self._n_layers += 1
        self._architecture["layer" + str(self._n_layers)] = Layer(n_neurons, activation_function)


class DeepNetwork:
    def __init__(self) -> None:
        pass

    def predict(self, X: np.array) -> np.array:
        pass

    def train_model(self, X: np.array, Y: np.array) -> None:
        pass

    def _propagate_forward(self) -> None:
        pass

    def _propagate_backward(self) -> None:
        pass

    