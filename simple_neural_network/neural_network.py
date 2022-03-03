from __future__ import annotations
from typing import Union
import numpy as np

from simple_neural_network.activation_function import ActivationFunction
from simple_neural_network.perceptron import Perceptron

class NeuralNetwork:
    def __init__(self, weights: np.ndarray, activation_function: ActivationFunction) -> None:
        """
            Neural networks which stores weights in matrix
                
            Representation of XOR operator:
            [
                [
                    [20, 20, -30], AND neuron
                    [20, 20, -10]  OR neuron
                ],
                [
                    [-60, 60, -30] OUTPUT neuron 
                ]
            ]

            Parameters:
                weights (np.ndarray 3 dimesions): weights and biases.
        """
        self.weights = weights
        self.activation_function = activation_function
    
    def update_weights(self, weights: np.ndarray) -> NeuralNetwork:
        """
            Instatiates a new NeuralNetwork object with updated weights.

            Parameters:
                weights (np.ndarray 3 dimesions): weights and biases.

            Returns:
                NeuralNetwork instance
        """
        return NeuralNetwork(weights, self.activation_function)

    def feed_forward(self, input_vector: np.ndarray) -> Union[int, float]:
        def contains_all_zeros(array: np.ndarray) -> bool:
            return np.all(array == 0)

        input_vector = np.append(input_vector, 1)  # 1 is for bias
        
        # Iterate layers. 
        for layer in self.weights:
            layer_output_vector = np.zeros(shape=len(layer.shape))
            
            # Iterate neurons.
            for neuron_index, neuron_weights in enumerate(layer):
                
                # If neuron is not active.
                if contains_all_zeros(neuron_weights):
                    layer_output_vector[neuron_index] = 0
                    continue

                # Compute perceptron.
                perceptron = Perceptron(np.array(neuron_weights), self.activation_function)
                layer_output_vector[neuron_index] = perceptron.fit_input(input_vector)
            input_vector = np.append(layer_output_vector, 1)

        output_layer = [not contains_all_zeros(neuron_weights) for neuron_weights in self.weights[-1]] + [False]  # False for bias.
        return input_vector[output_layer]

