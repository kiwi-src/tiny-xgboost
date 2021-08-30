import abc

import numpy as np


class Objective(abc.ABC):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def loss(self, labels, logits):
        pass

    @abc.abstractmethod
    def gradients(self, labels, logits):
        pass

    @abc.abstractmethod
    def hessians(self, labels, predictions):
        pass


class BinaryCrossentropy(Objective):

    def loss(self, labels, probs):
        loss = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
        return np.mean(loss)

    def gradients(self, labels, probs):
        # Derivative of -(y * log(1/(1+exp(-w))) + (1.0 - y) * log(1.0 - 1/(1+exp(-w)))) wrt w
        # probs = 1/(1+exp(-w))
        # XGBoost doesn't multiply with 1/len(labels)
        return probs - labels  # * 1/len(labels)

    def hessians(self, labels, probs):
        # Derivative of probs - labels wrt w
        # probs = 1/(1+exp(-w))
        # XGBoost doesn't multiply with 1/len(labels)
        return probs * (1 - probs)  # * 1/len(labels)


class SquaredError(Objective):

    def loss(self, labels, predictions):
        # In order to compare the loss of different datasets the mean is computed
        return 0.5 * np.mean(np.square(labels - predictions))

    def gradients(self, labels, logits):
        # XGBoost doesn't multiply with 1/len(labels)
        return -(labels - logits)  # * 1/len(labels)

    def hessians(self, labels, predictions):
        # XGBoost doesn't multiply with 1/len(labels)
        return np.full(len(labels), 1)  # * 1/len(labels)
