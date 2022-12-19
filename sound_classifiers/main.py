import os
import pickle

import torch
from torch.utils.data import DataLoader

from sound_classifiers.metrics import calc_metrics
from sound_classifiers.models import CNNModel, get_linear_svm, get_rbf_svm
from sound_classifiers.train import predict, train
from sound_classifiers.utils import collate_fn, set_random_seed
from sound_classifiers.working_with_data import (analyze_data, get_speakers,
                                                 load_data, preprocess_dataset)

DATA_PATH = '.'


def preprocess_data():
    if os.path.exists('preprocessed_data.pickle'):
        print('Data exists, skipping preprocessing.')
        return

    print('Loading data')

    train_dataset, test_dataset = load_data(DATA_PATH)
    analyze_data(DATA_PATH, train_dataset, test_dataset)

    print('Preprocessing data')
    x_train, y_train = preprocess_dataset(DATA_PATH, train_dataset)
    x_val, y_val = preprocess_dataset(DATA_PATH, test_dataset)

    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
    }

    with open('preprocessed_data.pickle', 'wb') as f:
        pickle.dump(data, f)


def load_preprocessed_data():
    with open('preprocessed_data.pickle', 'rb') as f:
        data = pickle.load(f)

    return data['x_train'], data['y_train'], data['x_val'], data['y_val']


def train_rbf_svm():
    set_random_seed(42)

    x_train, y_train, x_val, y_val = load_preprocessed_data()

    print('Training RBF SVM')
    rbf_svm = get_rbf_svm()
    rbf_svm = rbf_svm.fit(x_train, y_train)
    rbf_svm_metrics = calc_metrics(y_val, rbf_svm.predict(x_val), 'RBF SVM')

    with open('rbf_svm.pickle', 'wb') as f:
        pickle.dump(rbf_svm, f)

    print(rbf_svm_metrics)


def train_linear_svm():
    set_random_seed(42)

    x_train, y_train, x_val, y_val = load_preprocessed_data()

    print('Training linear SVM')
    linear_svm = get_linear_svm()
    linear_svm = linear_svm.fit(x_train, y_train)
    linear_svm_metrics = calc_metrics(
        y_val, linear_svm.predict(x_val), 'Linear SVM')

    with open('linear_svm.pickle', 'wb') as f:
        pickle.dump(linear_svm, f)

    print(linear_svm_metrics)


def train_cnn():
    set_random_seed(42)

    train_dataset, test_dataset = load_data(DATA_PATH)

    print('Training CNN')
    batch_size = 128
    speakers = get_speakers(DATA_PATH)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(
                                      batch, speakers, train=True))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=lambda batch: collate_fn(
                                     batch, speakers))
    model = CNNModel(n_output=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.7, verbose=True)
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    train(model, train_dataloader, test_dataloader,
          criterion, optimizer, device, 6, scheduler)

    _, pred, true = predict(model, test_dataloader, criterion, device)
    cnn_metrics = calc_metrics(true, pred, 'CNN')
    print(cnn_metrics)


def main():
    preprocess_data()
    train_rbf_svm()
    train_linear_svm()
    train_cnn()


if __name__ == '__main__':
    main()
