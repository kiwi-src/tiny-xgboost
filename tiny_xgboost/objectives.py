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
        return probs - labels

    def hessians(self, labels, probs):
        # Derivative of probs - labels wrt w
        # probs = 1/(1+exp(-w))
        return probs * (1 - probs)


class SquaredError(Objective):

    def loss(self, labels, predictions):
        return 0.5 * np.sum(np.square(labels - predictions))

    def gradients(self, labels, logits):
        return -(labels - logits)

    def hessians(self, labels, predictions):
        return np.full(len(labels), 1)
