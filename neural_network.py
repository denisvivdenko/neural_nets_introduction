import numpy as np
from typing import Tuple


class NeuralNetwork:
    def __init__(self, n_hidden_units: int) -> None:
        self.n_hidden_units = n_hidden_units
        self.parameters = dict()

    def predict(self, X: np.array) -> np.array:
        pass

    def train_model(self, X: np.array, Y: np.array) -> None:
        pass

    def forward_propagation(self, parameters: dict, X: np.array) -> Tuple[float, dict]:
        """Returns A[L] and cached values Z[l] and A[l] in dictionary."""
        pass

    def backward_propagation(self, cache: dict, parameters: dict, X: np.array, Y: np.array) -> dict:
        """Computes gradients for parameters and returns dictionary with dW1, db1, dW2, db2 values."""
        pass

    def update_parameters(self, parameters: dict, gradients: dict) -> dict:
        """Returns dict with updated parameters"""
        pass
    
    def initialize_parametres(self, n_input_units: int, n_hidden_units: int, n_output_units: int) -> dict:
        """Randomly initializes W1, b1, W2, b2 parameters and store them in a dictionary."""    
        parameters = dict()
        parameters["W1"] = np.random.randn(n_hidden_units, n_input_units) * 0.01
        parameters["W2"] = np.random.randn(n_output_units, n_hidden_units) * 0.01
        parameters["b1"] = 0
        parameters["b2"] = 0
        return parameters
    
    def sigma(self, Z: np.array) -> np.array:
        """Sigma activation function 1 / (1 + exp(-Z))."""
        return 1  / (1 + np.exp(-Z))