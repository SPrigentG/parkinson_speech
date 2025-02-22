import pandas as pd

from process import split_data, oversample_data, normalize_data, select_features


def main(path: str):
    df = pd.read_csv(path, header=[0,1], index_col=0)
    df.columns = df.columns.droplevel()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sm, y_train_sm = oversample_data(X_train, y_train)
    X_train_sc, X_test_sc = normalize_data(X_train_sm.iloc[:, 1:], X_test.iloc[:, 1:])
    X_train_sm = X_train_sm.astype(float)
    X_train_sm.iloc[:, 1:] = X_train_sc
    X_test = X_test.astype(float)
    X_test.iloc[:, 1:] = X_test_sc
    X_train_slct, X_test_slct = select_features(X_train_sc, X_test_sc)


if __name__ == '__main__':
    main()