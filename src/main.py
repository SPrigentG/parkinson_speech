import argparse

import pandas as pd
from omegaconf import OmegaConf
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from process import split_data, oversample_data, normalize_data, select_features


def __create_parser():
    """Parse user inputs."""
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-d', '--data_file',
                        action="store", type=str,
                        help='path to csv file with data',
                        default='./in/pd_speech_features_AdS.csv')
    parser.add_argument('-c', '--config_file',
                        action="store", type=str,
                        help='path to config file to use',
                        default='config.yaml')
    args = parser.parse_args()

    return args


def process_data(
        df: pd.DataFrame, target_col: str,
        binary_data_col: int,
        test_size: float, random_state: int,
        nb_of_features: int
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Take data and apply pre-processing.
    Pre-processing is a 4 step process that can only be applied to
    clean data. The process will split data in train and test datasets,
    then it will oversample the data using the SMOTE technique,
    then it will calculate and apply normalization to the train
    data (with the exception of specified features), apply
    normalization to the test dataset and finally select a 
    limited number of features based on the scoring on an
    F-test. This function will return a train dataset, a test dataset
    and their corresponding targets.

    Args:
        df (pd.DataFrame): original data
        target_col (str): name of the column in df containing the classes
        binary_data_col (int): name of features that are binary and should 
                               be excluded from normalization
        test_size (float): size of the fraction of df that should be kept for
                           model testing
        random_state (int): seed used for splitting the data in a reproducible manner
        nb_of_features (int): number of features to select in dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            processed train and test dataset, with their respective targets
    """
    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size, random_state)
    X_train_sm, y_train_sm = oversample_data(X_train, y_train)
    X_train_sc, X_test_sc = normalize_data(X_train_sm,
                                           X_test,
                                           binary_data_col)
    X_train_slct, X_test_slct = select_features(X_train_sc,
                                                X_test_sc,
                                                y_train_sm,
                                                nb_of_features)

    return X_train_slct, X_test_slct, y_train_sm, y_test


def main(args):
    """Load data, process it, trains 3 models and displays results"""
    conf = OmegaConf.load(args['config_file'])
    header = list(conf['header'])
    index = list(conf['index'])
    random_state = conf['random_state']

    df = pd.read_csv(args['data_file'], header=header, index_col=index)
    if len(header) > 1:
        df.columns = df.columns.droplevel()
    X_train, X_test, y_train, y_test = process_data(df,
                                                    conf['target_col'],
                                                    conf['binary_columns'],
                                                    conf['test_size'],
                                                    random_state,
                                                    conf['nb_of_features'])

    svc_clf = SVC()
    svc_clf.fit(X_train, y_train)
    y_pred_svc = svc_clf.predict(X_test)
    cm_svc = confusion_matrix(y_test, y_pred_svc)
    cm_plot_svc = ConfusionMatrixDisplay(cm_svc)
    cm_plot_svc.plot()
    print(classification_report(y_test, y_pred_svc))

    ada_clf = AdaBoostClassifier(n_estimators=100)
    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)
    cm_ada = confusion_matrix(y_test, y_pred_ada)
    cm_plot_ada = ConfusionMatrixDisplay(cm_ada)
    cm_plot_ada.plot()
    print(classification_report(y_test, y_pred_ada))

    mlp_clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64),
                            early_stopping=True)
    mlp_clf.fit(X_train, y_train)
    y_pred_mlp = mlp_clf.predict(X_test)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    cm_plot_mlp = ConfusionMatrixDisplay(cm_mlp)
    cm_plot_mlp.plot()
    print(classification_report(y_test, y_pred_mlp))


if __name__ == '__main__':
    args = __create_parser()

    main(vars(args))
