from typing import Tuple
import numpy as np

class LogisticRegression:
    def __init__(self, n_dimentions: int, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.weights = np.zeros(shape=(n_dimentions, 1))
        self.bias = 0
    
    def predict(self, X: np.array) -> np.array:
        predictions = np.dot(self.weights.T, X) + self.bias
        return np.array([1 if prediction > self.threshold else 0 for prediction in predictions])

    def train_model(self, X: np.array, y: np.array, n_iterations: int, learning_rate: float) -> None:
        pass
    
    def _sigma_function(self, Z: np.array) -> np.array:
        """Vectorized function: 1 / (1 + exp(-z))"""
        return 1 / (1 + np.exp(-Z))
    
    def _backpropagate_values(self, weights: np.array, bias: float, X: np.array, Y: np.array) -> Tuple[float, float]:
        """
        Returns derivatives:
            dw: np.array
            db: np.array 
        """
        pass

    def _optimize_parameters(self, weights: np.array, bias: float, n_iterations: int, learning_rate: float) -> Tuple[np.array, float]:
        pass