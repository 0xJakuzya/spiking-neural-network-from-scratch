import numpy as np
import config

class SNN():
    def __init__(self, n_channels=config.n_channels, n_neurons=config.n_neurons, fs=config.fs,
                    max_prob=config.max_prob, scale=config.scale):
        self.n_channels = n_channels
        self.n_inputs = n_channels * 2
        self.n_neurons = n_neurons
        self.fs = fs
        self.max_prob = max_prob
        self.scale = scale
        self.rng = np.random.default_rng(42)
        self.W = self.rng.uniform(0.18, 0.40, size=(self.n_inputs, self.n_neurons)).astype(np.float32)
        self.threshold = 1.0
        self.beta = np.exp(-1.0 / (fs * 0.05))
        self.theta_decay = np.exp(-1.0 / (fs * 0.05))
        self.A_plus = 0.0003
        self.A_minus = 0.00045
        self.trace_decay = np.exp(-1.0 / (fs * 0.02))
        self.w_min = 0.03
        self.w_max = 0.8
        self.reset_state()

    def encode(self, eeg):
        positive = np.maximum(eeg, 0.0)
        negative = np.maximum(-eeg, 0.0)
        amplitudes = np.concatenate([positive, negative], axis=-1)
        probabilities = self.max_prob * np.clip(amplitudes / self.scale, 0.0, 1.0)
        spikes = (self.rng.random(probabilities.shape) < probabilities).astype(np.float32)
        return spikes

    def reset_state(self):
        self.v = np.zeros(self.n_neurons, dtype=np.float32)
        self.theta = np.full(self.n_neurons, self.threshold, dtype=np.float32)
        self.pre_trace = np.zeros(self.n_inputs, dtype=np.float32)
        self.post_trace = np.zeros(self.n_neurons, dtype=np.float32)

    def stdp(self, pre_spikes, post_spikes):
        self.pre_trace *= self.trace_decay
        self.post_trace *= self.trace_decay
        dW_plus = self.A_plus * np.outer(self.pre_trace, post_spikes)
        dW_minus = self.A_minus * np.outer(pre_spikes, self.post_trace)
        self.W += dW_plus - dW_minus
        np.clip(self.W, self.w_min, self.w_max, out=self.W)
        self.pre_trace += pre_spikes
        self.post_trace += post_spikes

    def forward(self, eeg, learn=False):
        eeg = np.asarray(eeg, dtype=np.float32)
        self.reset_state()
        input_spikes = self.encode(eeg)
        T = input_spikes.shape[0]
        output_spikes = np.zeros((T, self.n_neurons), dtype=np.float32)
        voltages = np.zeros_like(output_spikes)
        for t in range(T):
            pre_spikes = input_spikes[t]
            current = pre_spikes @ self.W
            self.v = self.beta * self.v + current
            voltages[t] = self.v
            self.theta = (self.threshold + (self.theta - self.threshold) * self.theta_decay)
            post_spikes = (self.v >= self.theta).astype(np.float32)
            self.v[post_spikes == 1] = 0.0
            self.theta += self.threshold * post_spikes
            if learn:
                self.stdp(pre_spikes, post_spikes)
            output_spikes[t] = post_spikes
        return input_spikes, output_spikes, voltages