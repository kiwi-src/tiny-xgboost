import time

import numpy as np

import utils
from tiny_xgboost import objectives
from tiny_xgboost.learner import Learner

if __name__ == '__main__':
    large = True
    if large:
        train_inputs, test_inputs, train_labels, test_labels = utils.load_data()
    else:
        train_inputs = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        train_labels = np.asarray([1, 0, 1, 0])
        test_inputs = train_inputs
        test_labels = train_labels

    print("TRAINING")
    if large:
        n_estimators = 1
        objective = objectives.SquaredError()
        learner = Learner(objective, n_estimators=n_estimators, vectorized=False,
                          max_depth=6, base_score=0.5)
    else:
        n_estimators = 1
        objective = objectives.SquaredError()
        learner = Learner(objective, n_estimators=n_estimators, vectorized=False,
                          max_depth=3, base_score=0.5)

    start_time = time.time()
    learner.fit(train_set=(train_inputs, train_labels), eval_set=(test_inputs, test_labels))
    end_time = time.time()

    print("\nTREE")
    dump = learner.get_dump()[n_estimators - 1]
    print(dump)

    print("\nPREDICTIONS (EXAMPLE)")
    predictions = learner.predict(test_inputs)
    print(predictions[0:4])

    print("\nLOSS (TEST SET)")
    loss = objective.loss(test_labels, np.asarray(predictions))
    print(loss)

    print("\nRMSE (Test)")
    print(utils.rmse(predictions, test_labels))

    print("\nRMSE (Train)")
    predictions = learner.predict(train_inputs)
    print(utils.rmse(predictions, train_labels))

    print("\nEXECUTION TIME")
    print(end_time - start_time)
