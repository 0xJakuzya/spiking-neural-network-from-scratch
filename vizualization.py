import matplotlib as plt
import numpy as np

from preprocessing import eeg_to_spikes

def plot_eeg_spikes(signal):
    spikes = eeg_to_spikes(signal)
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(signal[:500])
    plt.title("EEG signal")
    plt.subplot(2, 1, 2)
    plt.eventplot([
        np.where(spikes[0, :500] == 1)[0],
        np.where(spikes[1, :500] == 1)[0]
    ])
    plt.title("Encoded spikes")
    plt.tight_layout()
    plt.show()

