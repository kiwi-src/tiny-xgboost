import unittest

import numpy as np

from tiny_xgboost.learner import Learner
from tiny_xgboost import objectives


class XgboostTest(unittest.TestCase):

    def test_regression_vectorized(self):
        self.regression_tests(vectorized=True)

    def test_regression_non_vectorized(self):
        self.regression_tests(vectorized=False)

    def test_classification_vectorized(self):
        self.classification_tests(vectorized=True)

    def test_classification_non_vectorized(self):
        self.classification_tests(vectorized=False)

    def test_n_estimators_vectorized(self):
        self.n_estimators_tests(True)

    def test_n_estimators_non_vectorized(self):
        self.n_estimators_tests(False)

    def compare(self, instances, labels,
                expected_loss, learner):
        learner.fit(instances, labels)
        predictions = learner.predict(instances)
        loss = learner.loss(labels, np.asarray(predictions))
        self.assertAlmostEqual(expected_loss, loss)

    def classification_tests(self, vectorized):
        objective = objectives.BinaryCrossentropy()

        instances = np.asarray([[1], [1], [2]])
        labels = np.asarray([1, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.4)
        self.compare(instances, labels, 0.11694405972957611, learner)

        instances = np.asarray([[1], [3], [1], [2]])
        labels = np.asarray([0, 1, 0, 1])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.12692803144454956, learner)

        instances = np.asarray([[1]])
        labels = np.asarray([0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.12692803144454956, learner)

        instances = np.asarray([
            [1, 3],
            [2, 3],
            [2, 4]
        ])
        labels = np.asarray([1, 0, 1])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.12692803144454956, learner)

        instances = np.asarray([[1.5], [0.1]])
        labels = np.asarray([1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.12692803144454956, learner)

        instances = np.asarray([[1], [2], [2]])
        labels = np.asarray([1, 0, 1])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.5044074753920237, learner)

    def regression_tests(self, vectorized):
        objective = objectives.SquaredError()

        instances = np.asarray([[1], [1], [2]])
        labels = np.asarray([1, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1], [2], [2]])
        labels = np.asarray([0, 1, 1])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1], [1]])
        labels = np.asarray([1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.125, learner)

        instances = np.asarray([[1], [1]])
        labels = np.asarray([0, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1], [0]])
        labels = np.asarray([1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1.0, 3.0], [2.0, 3.0], [2.0, 4.0]])
        labels = np.asarray([1, 0, 1])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1.0, 3.0], [2.0, 3.0], [2.0, 4.0], [3.0, 4.0]])
        labels = np.asarray([1, 0, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1.0], [2.0], [2.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0625, learner)

        instances = np.asarray(
            [[0.644, .247, -0.447], [0.385, 1.8, 1.037], [1.214, -0.166, 0.004]])
        labels = np.asarray([1, 0, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        instances = np.asarray([[1.0], [2.0], [3.0], [3.0]])
        labels = np.asarray([0.8, 0.2, 0.8, 0.8])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=6, base_score=0.5)
        self.compare(instances, labels, 0.0, learner)

        # depth 0
        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=0, base_score=0.5)
        self.compare(instances, labels, 0.125, learner)

        # depth 1
        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=1, base_score=0.5)
        self.compare(instances, labels, 0.08333333333333334, learner)

        # depth 2
        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        learner = Learner(objective, vectorized=vectorized, n_estimators=1,
                          max_depth=2, base_score=0.5)
        self.compare(instances, labels, 0.0625, learner)

    def n_estimators_tests(self, vectorized):
        # Squared error
        objective = objectives.SquaredError()
        learner = Learner(objective, vectorized=vectorized, n_estimators=2,
                          max_depth=6, base_score=0.5)
        instances = np.asarray([[1.0], [2.0], [3.0], [3.0]])
        labels = np.asarray([1, 0, 1, 1])
        self.compare(instances, labels, expected_loss=0.0, learner=learner)

        learner = Learner(objective, vectorized=vectorized, n_estimators=2,
                          max_depth=1, base_score=0.5)
        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        self.compare(instances, labels, expected_loss=0.06481481481481481, learner=learner)

        learner = Learner(objective, vectorized=vectorized, n_estimators=3,
                          max_depth=1, base_score=0.5)
        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        self.compare(instances, labels, expected_loss=0.02623456790123457, learner=learner)

        # Binary cross entropy
        learner = Learner(objectives.BinaryCrossentropy(), vectorized=vectorized, n_estimators=3,
                          max_depth=1, base_score=0.5)
        instances = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        labels = np.asarray([1, 0, 1, 0])
        self.compare(instances, labels, expected_loss=0.25448448949610347, learner=learner)


if __name__ == '__main__':
    unittest.main()
