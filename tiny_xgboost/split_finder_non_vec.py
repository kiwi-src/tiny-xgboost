import numpy as np

from tiny_xgboost.split_finder import SplitFinder


class SplitFinderNonVec(SplitFinder):

    @staticmethod
    def _calc_gain(gradient_sum, hessian_sum):
        if hessian_sum > 0.0:
            return np.square(gradient_sum) / hessian_sum
        else:
            return 0.0

    def find_split(self, instances, gradients, hessians):
        gradient_sum = gradients.sum()
        hessian_sum = hessians.sum()
        root_gain = self._calc_gain(gradient_sum, hessian_sum)
        sorted_instance_indices = instances.argsort(axis=0)
        best_feature_index = None
        best_split_value = None
        best_loss_change = 0.0
        best_split_index = None
        num_features = len(instances[0])

        for feature_index in range(num_features):
            gradient_sum_ge = 0
            hessian_sum_ge = 0
            split_index = 0
            last_feature_value = None

            for instance_index in reversed(sorted_instance_indices[:, feature_index]):
                feature_value = instances[instance_index, feature_index]

                if split_index == 0:
                    last_feature_value = feature_value
                    # Greater equal (value >= split value)
                    gradient_sum_ge += gradients[instance_index]
                    hessian_sum_ge += hessians[instance_index]
                    split_index += 1
                    continue

                if feature_value != last_feature_value and hessian_sum_ge >= self._min_child_weight:
                    # Lower than (value < split value)
                    gradient_sum_lt = gradient_sum - gradient_sum_ge
                    hessian_sum_lt = hessian_sum - hessian_sum_ge

                    if hessian_sum_lt >= self._min_child_weight:
                        loss_change = self._calc_gain(gradient_sum_ge, hessian_sum_ge) + \
                                      self._calc_gain(gradient_sum_lt, hessian_sum_lt) - \
                                      root_gain
                        proposed_split = (feature_value + last_feature_value) * 0.5

                        if loss_change > best_loss_change:
                            best_loss_change = loss_change
                            best_feature_index = feature_index
                            best_split_value = proposed_split
                            best_split_index = split_index

                gradient_sum_ge += gradients[instance_index]
                hessian_sum_ge += hessians[instance_index]
                last_feature_value = feature_value
                split_index += 1

        if best_loss_change <= self._epsilon:
            # If loss change is smaller or equal to 0.0 then no split is optimal
            # Epsilon is used instead of 0.0 because of floating point errors
            return None, None, None, None

        # Select only the indices of the best feature
        sorted_indices_best_feature = np.flip(sorted_instance_indices[:, best_feature_index])

        # Index of the instances where value >= split_value
        best_indices_lt = sorted_indices_best_feature[best_split_index:]

        # Index of the instances where value < split value
        best_indices_ge = sorted_indices_best_feature[:best_split_index]

        return best_feature_index, best_split_value, best_indices_lt, best_indices_ge
