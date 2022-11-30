import random

import numpy as np
import torch
import torchaudio


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(
        batch,
        speakers,
        train=False,
        crop_size=50000,
        resampled_freq=4000):
    tensors = []
    targets = []

    for waveform, f, _, _, s, _, _ in batch:
        w = torchaudio.transforms.Resample(f, resampled_freq)(waveform)
        if w.shape[1] > crop_size:
            st = np.random.randint(0, w.shape[1] - crop_size)
            w = w[:, st:]

        if train:
            # simple augmentation
            w += np.random.normal(0, torch.std(w).item() / 10, size=w.shape)

        tensors.append(waveform)
        g = int(speakers[s] == 'M')
        t = torch.LongTensor([0, 0])
        t[g] = 1
        targets.append(g)

    tensors = pad_sequence(tensors)
    targets = torch.LongTensor(targets)
    return tensors, targets
