import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

import config

class EEGProcessing():
    def __init__(self, fs=config.fs, low=config.low, high=config.high, window_size=config.window_size):
        self.fs = fs
        self.low = low
        self.high = high
        self.window_size = window_size
        self.sos = butter(N=4, Wn=[self.low, self.high], btype="bandpass", fs=self.fs, output="sos")

    def load_csv(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            channels = file.readline().strip().split(",")
            signals = np.loadtxt(file, delimiter=",", ndmin=2)
        time = np.arange(signals.shape[0]) / self.fs
        return time, signals, channels

    def filter_signals(self, signals):
        if signals.ndim != 2:
            print(f'eeg shape problem')
        return sosfiltfilt(self.sos, signals, axis=0)

    def create_windows(self, signals):
        n_samples = int(self.window_size * self.fs)
        windows = []
        for start in range(0, len(signals) - n_samples + 1, n_samples):
            end = start + n_samples
            windows.append(signals[start:end])
        return np.stack(windows).astype(np.float32)

    def normalize_signals(self, windows):
        windows = np.asarray(windows, dtype=np.float32)
        mean = windows.mean(axis=(0, 1), keepdims=True)
        std = windows.std(axis=(0, 1), keepdims=True)
        normalized = ((windows - mean) / np.maximum(std, 1e-8))
        return normalized