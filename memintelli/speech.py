"""Speech Commands GRU and the preprocessing stored with its checkpoint."""
import numpy as np
import torch
from torch import nn


class AudioGRU(nn.Module):
    def __init__(self, classes=35):
        super().__init__()
        self.gru = nn.GRU(40, 128, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


class SpeechFeatures:
    def __init__(self, checkpoint):
        import torchaudio
        self.config = checkpoint["features"]
        c = self.config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=c["sample_rate"], n_fft=c["n_fft"], win_length=c["win_length"],
            hop_length=c["hop_length"], n_mels=c["n_mels"], center=c["center"], power=c["power"],
        )
        self.mean, self.std = checkpoint["mean"].float(), checkpoint["std"].float()
        if self.mean.shape != (40,) or self.std.shape != (40,) or not (self.std > 0).all():
            raise ValueError("Invalid checkpoint feature normalization")

    def __call__(self, path):
        import soundfile
        waveform, rate = soundfile.read(str(path), dtype="float32")
        if rate != self.config["sample_rate"] or waveform.ndim != 1 or len(waveform) > self.config["samples"]:
            raise ValueError("Expected mono, 16 kHz audio of at most one second")
        waveform = np.pad(waveform, (0, self.config["samples"]-len(waveform)))
        feature = self.mel(torch.from_numpy(waveform)).clamp_min(self.config["log_floor"]).log().T
        return (feature-self.mean)/self.std
