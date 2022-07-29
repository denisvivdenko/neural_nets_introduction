import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from dataset.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


class NeuralNetwork:
    def __init__(self, n_hidden_units: int) -> None:
        self.n_hidden_units = n_hidden_units
        self.parameters = dict()

    def predict(self, X: np.array, parameters: dict = None) -> np.array:
        pass

    def train_model(self, X: np.array, Y: np.array, n_epoch: int = 1000, learning_rate: float = 0.05) -> None:
        n_input_units = X.shape[0]
        n_output_units = X.shape[0]
        parameters = self.initialize_parametres(n_input_units, self.n_hidden_units, n_output_units)
        for epoch in range(n_epoch):
            A_output, cache = self.forward_propagation(parameters, X)
            gradients = self.backward_propagation(cache, parameters, X, Y)
            parameters = self.update_parameters(parameters, gradients, learning_rate)
            if epoch % 1000 == 0: print(f"Cost: {self.compute_cost(A_output, Y)}")
        print(f"Final cost: {self.compute_cost(A_output, Y)}")
        self.parameters = parameters        

    def forward_propagation(self, parameters: dict, X: np.array) -> Tuple[float, dict]:
        """Returns A[L], Cost and cached values Z[l] and A[l] in dictionary."""
        W1, W2 = parameters["W1"], parameters["W2"]
        b1, b2 = parameters["b1"], parameters["b2"]
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigma(Z2)
        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }
        return A2, cache

    def backward_propagation(self, cache: dict, parameters: dict, X: np.array, Y: np.array) -> dict:
        """Computes gradients for parameters and returns dictionary with dW1, db1, dW2, db2 values."""
        m = X.shape[1]
        W1, W2 = parameters["W1"], parameters["W2"]  # W1.shape = (n1, nx); W2.shape = (n2, n1)
        A1, A2 = cache["A1"], cache["A2"]  # A1.shape = (n1, m); A2.shape = (n2, m)
        Z1, Z2 = cache["Z1"], cache["Z2"]  # Z1.shape = (n1, m); Z2.shape = (n2, m)
        dZ2 = A2 - Y  # (n2, m)
        dW2 = np.dot(dZ2, A1.T) / m  # (n2, n1)  
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # (n2, 1)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # (n1, m)
        dW1 = np.dot(dZ1, X.T) / m  # (n1, nx)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (n1, 1)
        gradients = {
            "dW1": dW1,
            "dW2": dW2,
            "db1": db1,
            "db2": db2
        }
        return gradients

    def update_parameters(self, parameters: dict, gradients: dict, learning_rate: float) -> dict:
        """Returns dict with updated parameters"""
        updated_parameters = {}
        W1, W2  = parameters["W1"], parameters["W2"]
        dW1, dW2 = gradients["dW1"], gradients["dW2"]
        db1, db2 = gradients["db1"], gradients["db2"]
        updated_parameters["W1"] = W1 - learning_rate * dW1
        updated_parameters["W2"] = W2 - learning_rate * dW2
        updated_parameters["b1"] = db1 - learning_rate * db1
        updated_parameters["b2"] = db2 - learning_rate * db2
        return updated_parameters

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

    def compute_cost(self, A_output: np.array, Y: np.array) -> float:
        m = A_output.shape[1]
        return -np.sum(Y * np.log(A_output) + (1 - Y) * np.log(1 - A_output)) / m


if __name__ == "__main__":
    X, Y = load_planar_dataset()
    model = NeuralNetwork(n_hidden_units=4)
    model.train_model(X, Y, n_epoch=10**5, learning_rate=0.1)
    plot_decision_boundary(lambda x: model.predict(x), X, Y)
    plt.show()