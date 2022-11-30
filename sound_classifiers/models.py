from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch import nn


def get_rbf_svm():
    return make_pipeline(StandardScaler(), SVC())


def get_linear_svm():
    return make_pipeline(StandardScaler(), LinearSVC(max_iter=5000))


# https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
class CNNModel(nn.Module):
    def __init__(
            self,
            n_output=35,
            stride=16,
            n_channel=32,
            reception_field=80):
        super().__init__()
        self.conv1 = nn.Conv1d(
            1, n_channel, kernel_size=reception_field, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = nn.functional.relu(self.bn4(x))
        x = self.pool4(x)
        x = nn.functional.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return nn.functional.softmax(x, dim=2)
