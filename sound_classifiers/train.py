import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


def train_one_epoch(
        model,
        train_dataloader,
        criterion,
        optimizer,
        device="cuda:0"):
    model.to(device).train()
    cum_loss = 0
    n_objects = 0
    for features, y in tqdm(train_dataloader):
        preds = model(features.to(device)).squeeze()
        loss = criterion(preds, y.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cum_loss += loss.detach().cpu().numpy() * features.shape[0]
        n_objects += features.shape[0]

    return cum_loss / n_objects


def predict(model, test_dataloder, criterion, device="cuda:0"):
    model.to(device).eval()
    with torch.no_grad():
        predicts = torch.tensor([])
        true_values = torch.tensor([])
        cum_loss = 0
        n_objects = 0
        for features, y in tqdm(test_dataloder):
            cur = model(features.to(device)).cpu().squeeze()
            predicts = torch.cat([predicts, np.argmax(cur, axis=1)])
            true_values = torch.cat([true_values, y])
            n_objects += features.shape[0]
            cum_loss += criterion(cur, y).item() * features.shape[0]

        return cum_loss / n_objects, np.array(predicts), np.array(true_values)


def train(
        model,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        device="cuda:0",
        n_epochs=10,
        scheduler=None):
    model.to(device)
    for epoch in range(n_epochs):
        print('Train')
        train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device)
        print('Evaluate')
        val_loss, predicted, true = predict(
            model, test_dataloader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        accuracy = accuracy_score(predicted, true)
        print('Epoch {}, val loss {:.3f}, train\
            loss {:.3f}, accuracy {:.3f}'.format(
            epoch + 1, val_loss, train_loss, accuracy))
