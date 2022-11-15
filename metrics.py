from sklearn.metrics import accuracy_score


def calc_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calc_metrics(y_true, y_pred, model_name):
    return {
        model_name + ' accuracy:': calc_accuracy(y_true, y_pred)
    }