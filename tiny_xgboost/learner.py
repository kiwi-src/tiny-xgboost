import numpy as np
import math
from tiny_xgboost.tree import Tree
from tiny_xgboost.objectives import BinaryCrossentropy, SquaredError


class Learner:

    def __init__(self, objective, vectorized, n_estimators, max_depth=6, base_score=0.5,
                 min_child_weight=0.0
                 ):
        self._objective = objective
        self._vectorized = vectorized
        self._n_estimators = n_estimators
        self._base_score = base_score
        self._max_depth = max_depth
        self._tree = None
        self._n_estimators = n_estimators
        self._trees = []
        self._epsilon = 1e-6
        self._min_child_weight = min_child_weight

    def fit(self, train_set, eval_set=None):
        instances_train = train_set[0]
        labels_train = train_set[1]
        predictions = np.full(len(labels_train), self._base_score)
        for i in range(self._n_estimators):
            tree = Tree(self._objective,
                        vectorized=self._vectorized,
                        max_depth=self._max_depth,
                        base_score=self._base_score,
                        epsilon=self._epsilon,
                        min_child_weight=self._min_child_weight
                        )
            tree.split(instances_train, labels_train, initial_predictions=predictions)
            self._trees.append(tree)
            predictions = self.predict(instances_train)
            train_loss = self._objective.loss(labels_train, predictions)

            if eval_set is None:
                print(f'[{i}] train_loss:{train_loss:.5f}')
            else:
                instances_eval = eval_set[0]
                labels_eval = eval_set[1]
                eval_loss = self._objective.loss(labels_eval, self.predict(instances_eval))
                print(f'[{i}] train_loss:{train_loss:.5f}\teval_loss:{eval_loss:.5f}')

    def predict(self, instances):
        if isinstance(self._objective, BinaryCrossentropy):
            # In this case probs are returned
            cum_logits = np.full(len(instances),
                                 math.log(self._base_score / (1.0 - self._base_score)))
            for tree in self._trees:
                logits = []
                for index, instance in enumerate(instances):
                    logits.append(tree.predict(instance))
                cum_logits = np.add(cum_logits, logits)

            def sigmoid(logits):
                return 1.0 / (1.0 + np.exp(-logits))
            return sigmoid(cum_logits)
        elif isinstance(self._objective, SquaredError):
            predictions = np.full(len(instances), self._base_score)
            for tree in self._trees:
                for index, instance in enumerate(instances):
                    prediction = tree.predict(instance)
                    predictions[index] += prediction
            return predictions
        else:
            raise NotImplementedError

    def get_dump(self):
        dump = []
        for tree in self._trees:
            dump.append(tree.get_dump())
        return dump

    def loss(self, labels, predictions):
        return self._objective.loss(labels, predictions)