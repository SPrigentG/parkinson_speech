import math
from random import randint
from sys import path

import pytest
import pandas as pd
import numpy as np
from collections import Counter

path.append('./src')

from process import (
    split_id, split_data, 
    oversample_data, 
    normalize_data, 
    select_features
)


@pytest.fixture
def get_df_data():
    data = {'feat1': [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1], 
            'feat2': [34, 11, 0, 86, 45, 6, 34, 11, 0, 86, 45, 6, 2, 34, 11, 0, 86, 45, 6, 34, 11, 0, 86, 45, 6, 2]}
    df_data = pd.DataFrame(data)
    return df_data


@pytest.fixture
def get_target():
    target = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], name='class')
    return target


# def test_split_size_stratify(get_df_data, get_target):
#     df = pd.concat([get_df_data, get_target], axis=1)
#     test_size = 0.2
#     _, _, y_train, y_test = split_data(df, test_size, None)
#     # train_counter = Counter(y_train)
#     # test_counter = Counter(y_test)
#     # train_value_count = list(train_counter.values())
#     # test_value_count = list(test_counter.values())
#     assert len(y_test) == math.ceil(len(get_target) * test_size)
#     # try:
#     #     assert round(train_value_count[0]/train_value_count[1], 1) == round(test_value_count[0]/test_value_count[1], 1)
#     # except:
#     #     assert round(train_value_count[1]/train_value_count[0], 1) == round(test_value_count[1]/test_value_count[0], 1)


def test_split_size(get_target):
    test_size = 0.2
    _, test_id = split_id(get_target, test_size, None)
    assert len(test_id) == math.ceil(len(get_target) * test_size)


def test_oversample_data(get_df_data, get_target):
    _, target_oversampled = oversample_data(get_df_data, get_target)
    data_counter = Counter(target_oversampled)

    all_equals = True
    test_val = list(data_counter.values())[0]
    for target in data_counter:
        if data_counter[target] != test_val:
            all_equals = False

    assert all_equals


def test_normalize(get_df_data, get_target):
    df = pd.concat([get_df_data, get_target], axis=1)
    X_train, X_test, _, _ = split_data(df, 0.2, None)
    X_train_sc, X_test_sc = normalize_data(X_train, X_test)

    random_feature = randint(0, len(get_df_data.columns))
    assert round(np.mean(X_train_sc[:, random_feature]), 1) == 0


