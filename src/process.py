import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def split_data(
        df: pd.DataFrame, target_col: str,
        test_size: float, random_state: int | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into a train dataset and a test dataset

    Args:
        df (pd.DataFrame): dataframe containing all observations and all features
        target_col (str): name of the column containing the target
        test_size (float): size of the fraction of df to use as test dataset
        random_state (int | None, optional): seed used to reproduce result
                                             Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            train dataset, test dataset, target for train, target for test
    """
    id_class = df[target_col]
    train_id, test_id = split_id(id_class, test_size, random_state)
    X_train = df.loc[train_id].sample(frac=1, random_state=random_state)
    X_test = df.loc[test_id].sample(frac=1, random_state=random_state)

    y_train = X_train[target_col]
    X_train.drop(columns=target_col, inplace=True)
    y_test = X_test[target_col]
    X_test.drop(columns=target_col, inplace=True)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def split_id(
        target: pd.Series, test_size: float,
        random_state: int | None = None
        ) -> tuple[pd.Series]:
    """Split id in order to keep replicates from an id together in same dataset

    Args:
        target (pd.Series): target with indexes from same patient
                            Warning: if id is lost, all observations will be
                            considered independent.
        test_size (float): size of the fraction of data to use as test dataset
        random_state (int | None, optional): seed used to reproduce result
                                             Defaults to None.

    Returns:
        tuple[pd.Series]: ids selected for train and test data
    """
    number_of_individuals = len(target)
    number_of_replicates = target.reset_index().groupby('index').count().iloc[0,0]
    if number_of_replicates > 1:
        target = target.iloc[range(0, number_of_individuals, number_of_replicates)]
    train_id, test_id = train_test_split(range(0, len(target)),
                                         test_size=test_size,
                                         random_state=random_state,
                                         stratify=target)

    return train_id, test_id

def oversample_data(
        imb_data: pd.DataFrame, target: pd.Series
        ) -> tuple[pd.DataFrame, pd.Series]:
    """Use SMOTE technique to oversample minority class

    Args:
        imb_data (pd.DataFrame): data that need to be oversampled
        target (pd.Series): class corresponding to data

    Returns:
        tuple[pd.DataFrame, pd.Series]: data with all classes having same
                                        number of observations.
    """
    smt = SMOTE()
    imb_data_sm, target_sm = smt.fit_resample(imb_data, target)

    return imb_data_sm, target_sm


def normalize_non_binary_data(
        train_data: pd.DataFrame, test_data: pd.DataFrame
        ) -> tuple[np.ndarray]:
    """Use Standard Scaler to normalize data

    Args:
        train_data (pd.DataFrame): train data from which to fit scaler
                                   and transform
        test_data (pd.DataFrame): test data to transform

    Returns:
        tuple[np.ndarray]: normalized train and test data
    """
    sc = StandardScaler()
    train_data_scaled = sc.fit_transform(train_data)
    test_data_scaled = sc.transform(test_data)

    return train_data_scaled, test_data_scaled


def normalize_data(
        train_data: pd.DataFrame, test_data: pd.DataFrame,
        binary_features_col: list[str] | str
        ) -> tuple[pd.DataFrame]:
    """Calculate and apply normalization to train dataset
    and apply it to test dataset.
    Features listed in binary_features_col will be excluded
    from normalization if they exist in the datasets.

    Args:
        train_data (pd.DataFrame): train dataset to calculate normalization
                                   from and apply to it.
        test_data (pd.DataFrame): test dataset where normalization will be
                                  only applied on.
        binary_features_col (list[str] | str): name of features to exclude
                                               from normalization

    Returns:
        tuple[pd.DataFrame]: normalized train and test datasets
    """
    try:
        train_binary_feat = train_data[binary_features_col]
        train_data.drop(columns=binary_features_col, inplace=True)
        test_binary_feat = test_data[binary_features_col]
        test_data.drop(columns=binary_features_col, inplace=True)
    except KeyError:
        train_binary_feat = pd.DataFrame()
        test_binary_feat = pd.DataFrame()

    train_data_sc, test_data_sc = normalize_non_binary_data(train_data,
                                                            test_data)

    train_data_sc = pd.DataFrame(train_data_sc, columns=train_data.columns)
    test_data_sc = pd.DataFrame(test_data_sc, columns=test_data.columns)

    train_data_sc = pd.concat([train_data_sc, train_binary_feat], axis=1)
    test_data_sc = pd.concat([test_data_sc, test_binary_feat], axis=1)

    return train_data_sc, test_data_sc


def select_features(
        train_data: pd.DataFrame, test_data: pd.DataFrame,
        train_target: pd.Series, nb_of_features: int = 50
        ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Reduce complexity of dataset by selecting best features based on 
    scoring on F-test.

    Args:
        train_data (pd.DataFrame): data on which selection will be based 
                                   and applied on
        test_data (pd.DataFrame): data where selection will also be applied
        train_target (pd.Series): class corresponding to the train_data
        nb_of_features (int, optional): number of features to keep. 
                                        Defaults to 50.

    Returns:
        tuple[pd.DataFrame]: train_data and test_data reduced to 
                             the number of features required,
                             name of features selected
    """
    if not isinstance(nb_of_features, int):
        nb_of_features = train_data.shape[1]

    selector = SelectKBest(score_func=f_classif, k=nb_of_features)
    train_slct = selector.fit_transform(train_data, train_target)
    test_slct = selector.transform(test_data)
    features_slct = selector.get_feature_names_out(train_data.columns)

    return train_slct, test_slct, features_slct
