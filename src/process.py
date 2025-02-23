import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def split_data(
        df: pd.DataFrame, test_size: float, random_state: int
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    id_class = df.iloc[:, -1]
    train_id, test_id = split_id(id_class, test_size, random_state)
    train = df.loc[train_id].sample(frac=1, random_state=random_state)
    test = df.loc[test_id].sample(frac=1, random_state=random_state)
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def split_id(target, test_size, random_state):
    number_of_individuals = len(target)
    number_of_replicates = target.reset_index().groupby('index').count().iloc[0,0]
    if number_of_replicates > 1:
        target = target.iloc[range(0, number_of_individuals, number_of_replicates)]
    train_id, test_id = train_test_split(range(0, len(target)),
                                         test_size=test_size,
                                         random_state=random_state,
                                         stratify=target)

    return train_id, test_id

def oversample_data(imb_data, target):
    smt = SMOTE()
    imb_data_sm, target_sm = smt.fit_resample(imb_data, target)

    return imb_data_sm, target_sm


def normalize_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    sc = StandardScaler()
    train_data_scaled = sc.fit_transform(train_data)
    test_data_scaled = sc.transform(test_data)

    return train_data_scaled, test_data_scaled


def select_features(train_data, test_data, train_target):
    selector = SelectKBest(score_func=f_classif, k=50)
    train_slct = selector.fit_transform(train_data, train_target)
    test_slct = selector.transform(test_data)

    return train_slct, test_slct