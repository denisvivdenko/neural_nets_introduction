from typing import Tuple
import numpy as np
import pandas as pd
import h5py


class LogisticRegression:
    def __init__(self, n_dimentions: int, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.weights = np.zeros(shape=(n_dimentions, 1))
        self.bias = 0
    
    def predict(self, X: np.array) -> np.array:
        predictions = np.dot(self.weights, X.T) + self.bias
        return np.array([1 if prediction > self.threshold else 0 for prediction in predictions])

    def train_model(self, X: np.array, Y: np.array, n_iterations: int, learning_rate: float) -> None:
        self.weights, self.bias = self._optimize_parameters(self.weights, self.bias, X, Y, n_iterations, learning_rate)
    
    def _sigma_function(self, Z: np.array) -> np.array:
        """Vectorized function: 1 / (1 + exp(-z))"""
        return 1 / (1 + np.exp(-Z))
    
    def _backpropagate_values(self, weights: np.array, bias: float, X: np.array, Y: np.array) -> Tuple[float, float]:
        """
        m - number of datapoints, n - number of features.
        weights.shape = (n, 1)
        X.shape = (m, n)
        Y.shape = (m, 1)

        Returns derivatives:
            dw: np.array (shape = (n, 1))
            db: float
        """
        n_records = X.shape[0]
        Z = np.dot(X, weights) + bias  # Z.shape = (m, 1)
        A = self._sigma_function(Z)  # A.shape = (m, 1)
        dw = np.dot(X.T, (A - Y)) / n_records  # dw.shape = (n, 1)
        db = np.sum((A - Y)) / n_records
        return (dw, db)

    def _optimize_parameters(self, weights: np.array, bias: float, X: np.array, Y: np.array, n_iterations: int, learning_rate: float) -> Tuple[np.array, float]:
        for _ in range(n_iterations):
            dw, db = self._backpropagate_values(weights, bias, X, Y)
            weights = weights - learning_rate * dw
            bias = bias - learning_rate * db
        return weights, bias


def read_h5(file_path: str) -> Tuple[np.array, np.array]:
    with h5py.File(file_path, "r") as file:
        X_key, Y_key = list(file.keys())[1: 3] 
        return file[X_key][()], file[Y_key][()]


if __name__ == "__main__":
    train_dataset_path = "./dataset/train_catvnoncat.h5"
    test_dataset_path = "./dataset/test_catvnoncat.h5"
    X_train, Y_train = read_h5(train_dataset_path)
    X_test, Y_test = read_h5(test_dataset_path)
    print(f"X_train.shape = {X_train.shape}. X_test.shape = {X_test.shape}\nY_train.shape = {Y_train.shape}. Y_test.shape = {Y_test.shape}")
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    Y_train, Y_test = Y_train.reshape(-1, 1).T, Y_test.reshape(-1, 1).T
    print("RESHAPING DATASET...")
    print(f"X_train.shape = {X_train.shape}. X_test.shape = {X_test.shape}\nY_train.shape = {Y_train.shape}. Y_test.shape = {Y_test.shape}")
