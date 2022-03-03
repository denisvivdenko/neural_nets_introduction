from abc import ABC
from typing import Union
import math
import numpy as np

class ActivationFunction(ABC):
    def get_result(self) -> Union[float, int]:
        raise Exception("Activation function is not defined.")

class ThresholdActivationFunction(ActivationFunction):
    def __init__(self, value: float) -> None:
        self.value = value

    def get_result(self) -> Union[float, int]:
        return 0 if self.value <= 0 else 1

class SigmoidActivationFunction(ActivationFunction):
    def __init__(self, value: float) -> None:
        self.value = value

    def get_result(self) -> Union[float, int]:
        return 1 / (1 + np.exp(-self.value)) 

if __name__ == "__main__":
    print(ThresholdActivationFunction(-1).get_result(), ThresholdActivationFunction(2).get_result())