import numpy as np

from simple_neural_network.neural_network import NeuralNetwork
from simple_neural_network.activation_function import ThresholdActivationFunction
from simple_neural_network.activation_function import SigmoidActivationFunction

if __name__ == "__main__":
    weights = np.array([[[20, 20, -30],
                        [20, 20, -10]],
                        [[-60, 60, -30],
                        [0, 0, 0]]])
    inputs = [np.array([0, 0]), np.array([1, 1]), np.array([1, 0]), np.array([0, 1])]
    neural_network = NeuralNetwork(weights=weights, activation_function=SigmoidActivationFunction)
    for input_vector in inputs:
        print(f"input: {input_vector} output: {neural_network.feed_forward(input_vector)[0]}")
    