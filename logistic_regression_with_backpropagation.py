from typing import Tuple
import numpy as np
import pandas as pd
import h5py


class LogisticRegression:
    def __init__(self, n_dimentions: int, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.weights = (.5 - np.random.rand(n_dimentions)).reshape(n_dimentions, 1)
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
        X.shape = (n, m)
        Y.shape = (1, m)

        Returns derivatives:
            dw: np.array (shape = (n, 1))
            db: float
        """
        n_records = X.shape[1]
        Z = np.dot(weights.T, X) + bias  # Z.shape = (1, m)
        A = self._sigma_function(Z)  # A.shape = (1, m)

        # print(A)
        cost_function = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        print(f"cost: {cost_function}")

        dw = np.dot(X, (A - Y).T) / n_records  # dw.shape = (n, 1)
        db = np.sum((A - Y)) / n_records
        return (dw, db)

    def _optimize_parameters(self, weights: np.array, bias: float, X: np.array, Y: np.array, n_iterations: int, learning_rate: float) -> Tuple[np.array, float]:
        for _ in range(n_iterations):
            dw, db = self._backpropagate_values(weights, bias, X, Y)
            weights = weights - learning_rate * dw
            bias = bias - learning_rate * db
            # print(f"weights: {weights} || bias: {bias}")
        return weights, bias


def read_h5(file_path: str) -> Tuple[np.array, np.array]:
    with h5py.File(file_path, "r") as file:
        X_key, Y_key = list(file.keys())[1: 3] 
        return file[X_key][()], file[Y_key][()]


def normalize_array(array: np.array) -> np.array:
    norm_vector = (np.linalg.norm(array, axis=1)).reshape(-1, 1)
    return array / norm_vector
    

if __name__ == "__main__":
    train_dataset_path = "./dataset/train_catvnoncat.h5"
    test_dataset_path = "./dataset/test_catvnoncat.h5"
    X_train, Y_train = read_h5(train_dataset_path)
    X_test, Y_test = read_h5(test_dataset_path)
    print(f"X_train.shape = {X_train.shape}. X_test.shape = {X_test.shape}\nY_train.shape = {Y_train.shape}. Y_test.shape = {Y_test.shape}")
    X_train = normalize_array(X_train.reshape(X_train.shape[0], -1).T)  # shape (n, m)
    X_test = normalize_array(X_test.reshape(X_test.shape[0], -1).T)  # shape (n, m_test)
    Y_train, Y_test = Y_train.reshape(-1, 1).T, Y_test.reshape(-1, 1).T  # shape (1, m), shape (1, m_test)
    print("RESHAPING DATASET...")
    print(f"X_train.shape = {X_train.shape}. X_test.shape = {X_test.shape}\nY_train.shape = {Y_train.shape}. Y_test.shape = {Y_test.shape}")

    model = LogisticRegression(n_dimentions=X_train.shape[0])
    model.train_model(X_train, Y_train, n_iterations=10000, learning_rate=0.1)