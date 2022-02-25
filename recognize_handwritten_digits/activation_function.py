from abc import ABC

class ActivationFunction(ABC):
    def fit_data(self, value: float) -> None:
        self.value = value

    def get_result(self):
        raise Exception("Activation function is not defined.")

class ThresholdActivationFunction(ActivationFunction):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def get_result(self):
        if self.thershold < self.value:
            return 0
        return 1