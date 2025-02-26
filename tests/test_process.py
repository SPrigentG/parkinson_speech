import math
from random import randint
from sys import path
from collections import Counter

import pytest
import pandas as pd
import numpy as np

path.append('./src')

from process import (
    split_id, split_data, 
    oversample_data, 
    normalize_data, 
    select_features
)


# TODO: this script only test the happy path, so it should also
# make sure that if bad data is given, it may at least show explicit
# errors to the user.

@pytest.fixture
def get_df_data():
    """Create a simple dataset for testing.
    TODO: no hard coding."""
    data = {'feat1': [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
                       0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
            'feat2': [34, 11, 0, 86, 45, 6, 34, 11, 0, 86, 45, 6, 2,
                      34, 11, 0, 86, 45, 6, 34, 11, 0, 86, 45, 6, 2]}
    df_data = pd.DataFrame(data)
    return df_data


@pytest.fixture
def get_target():
    """Create class for a fake dataset.
    TODO: take get_df_data, make class size according to size of df.
    TODO: control imbalance."""
    target = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                        1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        name='class')
    return target


def test_split_size(get_target):
    """Test if size of test dataset is of the expected size.
    TODO: control test size var."""
    test_size = 0.2
    _, test_id = split_id(get_target, test_size, None)
    assert len(test_id) == math.ceil(len(get_target) * test_size)


def test_oversample_data(get_df_data, get_target):
    """Test if minority class is of same size than majority class
    after oversampling."""
    _, target_oversampled = oversample_data(get_df_data, get_target)
    data_counter = Counter(target_oversampled)

    all_equals = True
    test_val = list(data_counter.values())[0]
    for target in data_counter:
        if data_counter[target] != test_val:
            all_equals = False

    assert all_equals


def test_normalize(get_df_data, get_target):
    """Test is any feature is taken, is mean equal to 0 after
    normalization.
    TODO: test if id are not repeated before erasing id,
    so that patient cannot be in both train and test
    dataset."""
    df = pd.concat([get_df_data, get_target], axis=1)
    X_train, X_test, _, _ = split_data(df, get_target.name, 0.2, None)
    X_train_sc, _ = normalize_data(X_train, X_test, '')

    random_feature = randint(0, get_df_data.shape[1])
    assert round(np.mean(X_train_sc.iloc[:, random_feature]), 1) == 0


def test_select(get_df_data, get_target):
    """Test if after feature selection, train dataset
    and test dataset have same size."""
    df = pd.concat([get_df_data, get_target], axis=1)
    X_train, X_test, y_train, _ = split_data(df, get_target.name, 0.2, None)
    X_train_slct, X_test_slct, _ = select_features(X_train,
                                                   X_test,
                                                   y_train,
                                                   1)
    assert X_train_slct.shape[1] == X_test_slct.shape[1]
