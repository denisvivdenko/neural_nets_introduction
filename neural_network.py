import numpy as np
from typing import Tuple


class NeuralNetwork:
    def __init__(self, n_hidden_units: int) -> None:
        self.n_hidden_units = n_hidden_units
        self.parameters = dict()

    def predict(self, X: np.array) -> np.array:
        pass

    def train_model(self, X: np.array, Y: np.array, n_epoch: int = 100, learning_rate: float = 0.05) -> None:
        n_input_units = X.shape[0]
        n_output_units = X.shape[0]
        parameters = self.initialize_parametres(n_input_units, self.n_hidden_units, n_output_units)
        for epoch in range(n_epoch):
            A_output, cost, cache = self.forward_propagation(parameters, X)
            gradients = self.backward_propagation(cache, parameters)
            parameters = self.update_parameters(parameters, gradients)
            if epoch % 10 == 0: print(f"Cost: {cost}")
        self.parameters = parameters        

    def forward_propagation(self, parameters: dict, X: np.array) -> Tuple[float, float, dict]:
        """Returns A[L], Cost and cached values Z[l] and A[l] in dictionary."""
        W1, W2 = parameters["W1"], parameters["W2"]
        b1, b2 = parameters["b1"], parameters["b2"]
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigma(Z2)
        return A2

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