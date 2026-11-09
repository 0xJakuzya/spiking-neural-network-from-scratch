import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from preprocessing import EEGProcessing
import config

class EEGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]

class EEGDataPipeline():
    def __init__(self, data=config.data, batch_size=config.batch_size, channel_names=None):
        self.data = Path(data)
        self.batch_size = batch_size
        self.channel_names = (None if channel_names is None else tuple(name.upper() for name in channel_names))
        self.seed = 42
        self.preprocessor = EEGProcessing()

    def split_subjects(self, data):
        open_subjects = {p.stem for p in (data / "open").glob("*.csv")}
        closed_subjects = {p.stem for p in (data / "closed").glob("*.csv")}
        subjects = np.array(sorted(open_subjects))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]
        return train_subjects, val_subjects, test_subjects

    def prepare_data(self, data, subjects, preprocessor):
        X = []
        y = []
        for subject in tqdm(subjects, desc="processing eeg", unit="subject"):
            for state, label in [("open", 0), ("closed", 1)]:
                path = data / state / f"{subject}.csv"
                _, signals, channels = preprocessor.load_csv(path)
                if self.channel_names is not None:
                    normalized_channels = [name.rstrip(".").upper() for name in channels]
                    channel_indices = [normalized_channels.index(name) for name in self.channel_names]
                    signals = signals[:, channel_indices]
                filtered = preprocessor.filter_signals(signals)
                windows = preprocessor.create_windows(filtered)
                X.append(windows)
                labels = np.full(len(windows), label, dtype=np.int64)
                y.append(labels)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        return X, y

    def split_data(self):
        train_subjects, val_subjects, test_subjects = self.split_subjects(self.data)

        X_train, y_train = self.prepare_data(self.data, train_subjects, self.preprocessor)
        self.train_dataset = EEGDataset(X_train, y_train)

        X_val, y_val = self.prepare_data(self.data, val_subjects, self.preprocessor)
        self.val_dataset = EEGDataset(X_val, y_val)

        X_test, y_test = self.prepare_data(self.data, test_subjects, self.preprocessor)
        self.test_dataset = EEGDataset(X_test, y_test)

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader