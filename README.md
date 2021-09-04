# Tiny XGBoost
Tiny XGBoost is a tiny Python implementation of the XGBoost C++ library.

Implementation goals:
* Simple Python implementation for research purposes
* Exactly match the results of XGBoost
* Make it fast by using vectorization

## Results
For comparing the performance of XGBoost with tiny-xgboost the regression datasets from LightGBM 
were used:
https://github.com/Microsoft/LightGBM/tree/master/examples/regression

|  | RMSE (Train) | RMSE (Test) | Execution Time
| ------------- | ------------- | ------------- | ------------- |
| XGBoost (C++) | 0.43059 | 0.44484 | 0.069
| tiny-xgboost (non-vectorized) | 0.43059 | 0.44484 | 6.361
| tiny-xgboost (vectorized) | 0.43059 | 0.44484 | 0.1314

The results can be reproduced by running the command
```
python run_tiny_xgboost.py
```

For the comparison the following arguments were used:
```
n_estimators = 1
max_depth = 6
```

## Reference
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD 
Conference on Knowledge Discovery and Data Mining, 2016