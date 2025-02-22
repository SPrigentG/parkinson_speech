import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    id_class = df.iloc[:, -1]
    number_of_individuals = len(id_class)
    id_class = id_class.iloc[range(0, len(number_of_individuals), 3)]
    train_id, test_id = train_test_split(range(0, number_of_individuals),
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=id_class)
    train = df.loc[train_id].sample(frac=1)
    test = df.loc[test_id].sample(frac=1)
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


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