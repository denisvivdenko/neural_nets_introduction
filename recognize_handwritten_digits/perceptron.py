import numpy as np

from recognize_handwritten_digits.activation_function import ActivationFunction

class Perceptron:
    def __init__(self, weight: np.array, activation_function: ActivationFunction) -> None:
        self.weights = weight
        self.activation_function = activation_function

    def fit_input(self, values: np.array) -> None:
        self.values = values
    
    def get_result(self) -> int:
        value = np.sum(self.weights * self.values)
        self.activation_function.fit_data(value=value)
        return self.activation_function.get_result()
    