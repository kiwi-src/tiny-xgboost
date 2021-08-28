import numpy as np


class SplitFinder:

    def __init__(self, epsilon, min_child_weight):
        self._epsilon = epsilon
        self._min_child_weight = min_child_weight

    def calc_weight(self, gradients, hessians):
        sum_gradients = np.sum(gradients)
        sum_hessians = np.sum(hessians)
        if sum_hessians < self._min_child_weight or sum_hessians <= 0.0:
            return 0.0
        else:
            return -sum_gradients / sum_hessians
