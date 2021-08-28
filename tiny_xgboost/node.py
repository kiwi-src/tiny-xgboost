class Node:

    def __init__(self, weight, max_depth):
        self.left_child = None
        self.right_child = None
        self.split_value = None
        self.feature_index = None
        self.is_leaf = None
        self.weight = weight
        self._max_depth = max_depth

    def __str__(self):
        if self.is_leaf:
            return f'leaf={self.weight}'
        else:
            return f'[feature_{self.feature_index}_value<{self.split_value}]'

    def split(self, split_finder, instances, gradients, hessians, vectorized, depth):

        if depth == self._max_depth:
            self.is_leaf = True
            return

        if vectorized:
            feature_index, split_value, best_indexes_lt, best_indexes_ge = \
                split_finder.find_split(instances, gradients, hessians)
        else:
            feature_index, split_value, best_indexes_lt, best_indexes_ge = \
                split_finder.find_split(instances, gradients, hessians)

        if split_value is None:
            self.is_leaf = True
            return

        else:
            # Init new node
            self.split_value = split_value
            self.feature_index = feature_index
            left_weight = split_finder.calc_weight(gradients[best_indexes_lt],
                                                   hessians[best_indexes_lt])
            self.left_child = Node(left_weight, self._max_depth)

            right_weight = split_finder.calc_weight(gradients[best_indexes_ge],
                                                    hessians[best_indexes_ge])
            self.right_child = Node(right_weight, self._max_depth)

        # value < split_value
        self.left_child.split(
            split_finder,
            instances[best_indexes_lt], gradients[best_indexes_lt], hessians[best_indexes_lt],
            vectorized=vectorized,
            depth=depth + 1,
        )

        # value >= split_value
        self.right_child.split(
            split_finder,
            instances[best_indexes_ge], gradients[best_indexes_ge], hessians[best_indexes_ge],
            vectorized=vectorized,
            depth=depth + 1)

    def predict(self, instance):
        if self.is_leaf:
            return self.weight
        elif instance[self.feature_index] < self.split_value:
            if self.left_child:
                return self.left_child.predict(instance)
        else:
            if self.right_child:
                return self.right_child.predict(instance)
