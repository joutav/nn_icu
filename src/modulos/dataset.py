from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pandas as pd


class Uti(Dataset):
    def __init__(self, x, y):
        dado = pd.read_csv(x)
        self._quantidade_features = dado.shape[1]
        self.features = dado.to_numpy()
        self.labels = pd.read_csv(y).to_numpy()
        

    def __getitem__(self, index):
        features = self.features[index]
        labels = self.labels[index]

        features = torch.from_numpy(features.astype(np.float32))
        labels = torch.from_numpy(labels)


        return features, labels

    def __len__(self):
        return len(self.features)

    @property
    def quantidade_features(self):
        return self._quantidade_features
