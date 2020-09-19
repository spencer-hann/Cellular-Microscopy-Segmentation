import numpy as np

from scipy.stats import entropy


class PseudoLabel:
    @staticmethod
    def entropy(array):
        axes = tuple(range(array.ndim))  # use all axes
        return entropy(array, axis=axes, base=2)

    @staticmethod
    def negative_mean_entropy(array):
        e = PseudoLabel.entropy(array)
        return -e / array.size

