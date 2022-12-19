import argparse
import os

from sound_classifiers.main import (preprocess_data, train_cnn,
                                    train_linear_svm, train_rbf_svm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action')

    return parser.parse_args()


def run_airflow():
    os.system('airflow dags test sound_classifiers')


if __name__ == '__main__':
    args = parse_args()

    actions = {
        'airflow': run_airflow,
        'preprocess_data': preprocess_data,
        'train_cnn': train_cnn,
        'train_linear_svm': train_linear_svm,
        'train_rbf_svm': train_rbf_svm,
    }

    actions[args.action]()
