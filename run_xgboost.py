import time
import xgboost as xgb

import utils

if __name__ == '__main__':
    model_type = xgb.XGBRegressor

    kwargs = {
        'n_estimators': 1,
        'max_depth': 6,
        'colsample_bytree': 1.0,
        'colsample_bynode': 1.0,
        'colsample_bylevel': 1.0,
        'missing': -1,
        'use_label_encoder': False,
        'reg_lambda': 0,
        'alpha': 0,
        'min_child_weight': 0,
        'base_score': 0.5,
        'learning_rate': 1,
        'tree_method': 'exact',
        'booster': 'gbtree',
        'nthread': 1,
    }

    if model_type == xgb.XGBRegressor:
        kwargs['objective'] = 'reg:squarederror'
    elif model_type == xgb.XGBClassifier:
        kwargs['objective'] = 'binary:logistic'

    model = model_type(**kwargs)
    train_inputs, test_inputs, train_labels, test_labels = utils.load_data()

    start_time = time.time()
    model.fit(train_inputs, train_labels,
              eval_set=[[test_inputs, test_labels]],
              verbose=True)
    end_time = time.time()
    best_iteration = model.get_booster().best_iteration

    booster = model.get_booster()
    dump = booster.get_dump(fmap='')[best_iteration]
    print(dump)

    print("\nRMSE (Test)")
    predictions = model.predict(test_inputs)
    print(utils.rmse(predictions, test_labels))

    print("\nRMSE (Train)")
    predictions = model.predict(train_inputs)
    print(utils.rmse(predictions, train_labels))

    print("\nEXECUTION TIME")
    print(end_time - start_time)