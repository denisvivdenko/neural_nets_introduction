import numpy as np

from simple_neural_network.activation_function import ActivationFunction

class Perceptron:
    def __init__(self, weight: np.array, activation_function: ActivationFunction) -> None:
        """
            Example:
                activation_function = BiasActivationFunction(bias=-3)
                weights = [-20, -30]
                perceptron = Perceptron(weights, activation_function)
                perceptron.fit_input(input_vector)
        """

        self.weights = weight
        self.activation_function = activation_function

    def fit_input(self, input_vector: np.array) -> float:
        """
            Returns perceptron output.
        """
        dot_product = np.sum(self.weights * input_vector)
        return self.activation_function(value=dot_product).get_result()
        
