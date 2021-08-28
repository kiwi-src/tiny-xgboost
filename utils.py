import numpy as np
import pandas as pd
import os


def load_data(data_dir='./data'):
    train_df = pd.read_csv(os.path.join(data_dir, 'regression.train'), header=None, sep='\t')
    test_df = pd.read_csv(os.path.join(data_dir, 'regression.test'), header=None, sep='\t')
    labels_train = train_df[0].values
    labels_test = test_df[0].values
    inputs_train = train_df.drop(0, axis=1).values
    inputs_test = test_df.drop(0, axis=1).values
    return inputs_train, inputs_test, labels_train, labels_test


def rmse(predictions, labels):
    return np.sqrt(np.mean(np.square(predictions - labels)))