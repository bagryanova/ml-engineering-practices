from .working_with_data import load_data, analyze_data, preprocess_dataset, get_speakers
from .models import get_RBF_SVM, get_linear_SVM, CNN_Model
from .metrics import calc_metrics
from .utils import set_random_seed
import torch
from torch.utils.data import DataLoader
from .utils import collate_fn
from .train import train, predict


def main():
    path = '.'
    set_random_seed(42)

    print('Loading data')
    train_dataset, test_dataset = load_data(path)
    analyze_data(path, train_dataset, test_dataset)

    print('Preprocessing data')
    X_train, y_train = preprocess_dataset(path, train_dataset)
    X_val, y_val = preprocess_dataset(path, test_dataset)

    print('Training RBF SVM')
    rbf_svm = get_RBF_SVM()
    rbf_svm = rbf_svm.fit(X_train, y_train)
    rbf_svm_metrics = calc_metrics(y_val, rbf_svm.predict(X_val), 'RBF SVM')
    print(rbf_svm_metrics)

    print('Training linear SVM')
    linear_svm = get_linear_SVM()
    linear_svm = linear_svm.fit(X_train, y_train)
    linear_svm_metrics = calc_metrics(y_val, linear_svm.predict(X_val), 'Linear SVM')
    print(linear_svm_metrics)

    print('Training CNN')
    batch_size = 128
    speakers = get_speakers(path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, speakers, train=True))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, speakers))
    model = CNN_Model(n_output=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7, verbose=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    train(model, train_dataloader, test_dataloader, criterion, optimizer, device, 6, scheduler)

    _, pred, true = predict(model, test_dataloader, criterion, device)
    cnn_metrics = calc_metrics(true, pred, 'CNN')
    print(cnn_metrics)


if __name__ == '__main__':
    main()
