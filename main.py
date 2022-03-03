import numpy as np

from simple_neural_network.neural_network import NeuralNetwork
from simple_neural_network.activation_function import ThresholdActivationFunction
from simple_neural_network.activation_function import SigmoidActivationFunction

if __name__ == "__main__":
    weights = np.array([
                [[20, 20, -30],
                 [20, 20, -10]],
                [[-60, 60, -30],
                 [0, 0, 0]]
            ], dtype=object)
    input_vector_1 = np.array([1, 0])
    input_vector_2 = np.array([1, 1])
    input_vector_3 = np.array([0, 1])
    input_vector_4 = np.array([0, 0])
    neural_network = NeuralNetwork(weights=weights, activation_function=SigmoidActivationFunction)
    print(neural_network.fit_input(input_vector_1), "\n")
    print(neural_network.fit_input(input_vector_2), "\n")
    print(neural_network.fit_input(input_vector_3), "\n")
    print(neural_network.fit_input(input_vector_4), "\n")


    