import numpy as np

from tiny_xgboost.node import Node
from tiny_xgboost.split_finder_non_vec import SplitFinderNonVec
from tiny_xgboost.split_finder_vec import SplitFinderVec


class Tree:

    def __init__(self, objective, vectorized, base_score, max_depth, epsilon,
                 min_child_weight):
        self._objective = objective
        self._base_score = base_score
        self._max_depth = max_depth
        self._root = None
        self._vectorized = vectorized
        self._epsilon = epsilon
        self._min_child_weight = min_child_weight

    def split(self, instances, labels, initial_predictions):
        gradients = self._objective.gradients(labels, initial_predictions)
        hessians = self._objective.hessians(labels, initial_predictions)

        if self._vectorized is True:
            split_finder = SplitFinderVec(epsilon=self._epsilon,
                                          min_child_weight=self._min_child_weight)
        else:
            split_finder = SplitFinderNonVec(epsilon=self._epsilon,
                                             min_child_weight=self._min_child_weight)

        self._root = Node(self._max_depth)
        self._root.split(split_finder, instances, gradients, hessians, self._vectorized, depth=0)

    def predict(self, instance):
        return self._root.predict(instance)

    def get_dump(self):
        return '\n'.join(self._get_dump(self._root, depth=0, end=[]))

    @staticmethod
    def _vertical_lines(end):
        vertical_lines = []
        for e in np.roll(end, 1)[1:]:
            if e:
                vertical_lines.append('    ')
            else:
                vertical_lines.append('\u2502' + ' ' * 3)
        return ''.join(vertical_lines)

    @staticmethod
    def _horizontal_line(last_node):
        if last_node:
            return '\u2514\u2500\u2500'
        else:
            return '\u251c\u2500\u2500'

    def _get_dump(self, node, depth, end):
        dump = []
        if depth > 0:
            indent = self._vertical_lines(end) + self._horizontal_line(end[-1])
            dump.append(f'{indent} {node}')
        else:
            dump.append(f'{node}')
        if node.left_child is not None:
            dump.extend(self._get_dump(node.left_child, depth + 1, end + [False]))
        if node.right_child is not None:
            dump.extend(self._get_dump(node.right_child, depth + 1, end + [True]))
        return dump
