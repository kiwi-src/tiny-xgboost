import numpy as np

from tiny_xgboost.split_finder import SplitFinder


class SplitFinderVec(SplitFinder):

    @staticmethod
    def _calc_gain(gradient_sum, hessian_sum):
        return np.divide(np.square(gradient_sum), hessian_sum,
                         out=np.zeros_like(gradient_sum),
                         where=hessian_sum > 0)

    def find_split(self, instances, gradients, hessians):
        num_instances = len(instances)
        num_features = len(instances[0])
        gradient_sum = gradients.sum()
        hessian_sum = hessians.sum()
        root_gain = self._calc_gain(gradient_sum, hessian_sum)

        # Sort instances in a descending order
        sorted_instance_indices = instances.argsort(axis=0)
        sorted_instance_indices = np.flip(sorted_instance_indices, axis=0)
        instances_flatten = instances.flatten(order='F')

        # Compute feature values and last feature values
        tmp = sorted_instance_indices + (np.arange(num_features) * num_instances)
        feature_values = instances_flatten[tmp[1:].flatten(order='F')]
        last_feature_values = instances_flatten[tmp[:-1].flatten(order='F')]

        # Greater equal
        gradient_sum_ge = np.cumsum(gradients[sorted_instance_indices[:-1]], axis=0).flatten(
            order='F')
        hessian_sum_ge = np.cumsum(hessians[sorted_instance_indices[:-1]], axis=0).flatten(
            order='F')

        # Lower than
        gradient_sum_lt = gradient_sum - gradient_sum_ge
        hessian_sum_lt = hessian_sum - hessian_sum_ge
        loss_changes = self._calc_gain(gradient_sum_ge, hessian_sum_ge) + \
                       self._calc_gain(gradient_sum_lt, hessian_sum_lt) - \
                       root_gain

        # Consider loss change only when hessian_sum_lt >= min_child_weight
        loss_changes = np.where(np.less(hessian_sum_lt, self._min_child_weight), self._epsilon,
                                loss_changes)

        # Consider loss change only when hessian_sum_gt >= min_child_weight
        loss_changes = np.where(np.less(hessian_sum_ge, self._min_child_weight), self._epsilon,
                                loss_changes)

        # Consider loss change only when feature_value != last_feature_value
        loss_changes = np.where(np.invert(np.not_equal(feature_values, last_feature_values)),
                                self._epsilon,
                                loss_changes)

        if len(loss_changes) == 0:
            return None, None, None, None

        best_loss_change_index = np.argmax(loss_changes)
        best_feature_index, best_split_index = np.unravel_index(best_loss_change_index, (
            num_features, num_instances - 1))

        best_loss_change = loss_changes[best_loss_change_index]
        if last_feature_values[best_split_index] is None:
            best_split_value = None
        else:
            best_split_value = (feature_values[best_loss_change_index] + last_feature_values[
                best_loss_change_index]) * 0.5

        if best_loss_change <= self._epsilon:
            # If loss change is smaller or equal to 0.0 then no split is optimal
            # Epsilon is used instead of 0.0 because of floating point errors
            return None, None, None, None

        # Select only the indices of the best feature
        sorted_indices_best_feature = sorted_instance_indices[:, best_feature_index]

        # Index of the instances where value >= split_value
        best_indices_lt = sorted_indices_best_feature[best_split_index + 1:]

        # Index of the instances where value < split value
        best_indices_ge = sorted_indices_best_feature[:best_split_index + 1]

        return best_feature_index, best_split_value, best_indices_lt, best_indices_ge
