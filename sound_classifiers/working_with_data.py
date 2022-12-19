import os

import librosa
import numpy as np
import pandas as pd
import torchaudio


def load_data(path):
    train = torchaudio.datasets.LIBRITTS(
        path, url="train-clean-100", download=True)
    test = torchaudio.datasets.LIBRITTS(path, url="test-clean", download=True)
    return train, test


def get_speakers(path):
    speakers = pd.read_csv(os.path.join(
        path, "LibriTTS/speakers.tsv"), sep='\t')
    speakers = speakers[['READER', 'GENDER']]
    return speakers.to_dict().get('READER')


def preprocess_dataset(path, dataset):
    speakers = get_speakers(path)

    x = [np.mean(librosa.feature.mfcc(y=np.array(d[0][0]), sr=d[1]), axis=1)
         for d in dataset]
    x = list(x)

    y = list([0 if speakers.get(d[4]) == 'F' else 1 for d in dataset])
    return x, y


def analyze_data(path, train_dataset, test_dataset):
    df = pd.read_csv(os.path.join(path, "LibriTTS/speakers.tsv"), sep='\t')
    train_speakers = set()
    test_speakers = set()

    for row in df.iloc:
        if row['SUBSET NAME'] == 'train-clean-100':
            train_speakers.add(row['READER'])
        if row['SUBSET NAME'] == 'test-clean':
            test_speakers.add(row['READER'])

    # ensure test dataset does not contain train speakers to avoid overtraining
    if len(train_speakers.intersection(test_speakers)) > 0:
        print("Warning: Train and test datasets contain the same speakers")

    speakers = get_speakers(path)
    for i in range(len(train_dataset)):
        if speakers.get(train_dataset[i][4]) is None:
            print(
                "Warning: Not every speaker has information\
                 about gender in train dataset")

    for i in range(len(test_dataset)):
        if speakers.get(test_dataset[i][4]) is None:
            print(
                "Warning: Not every speaker has information\
                 about gender in test dataset")
