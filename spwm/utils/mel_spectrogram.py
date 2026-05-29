"""
Mel-Spectrogram Extractor (Fall-Mamba Optimization 5)

Lightweight audio processing for edge deployment.
Replaces heavy WavJEPA (~200M) with Mel-Spectrogram + lightweight CNN
when running on Jetson or CPU-only edge devices.

From Fall-Mamba: Mel-Spectrogram → 2D image-like representation → CNN
Works with synchronized 25fps video (Le2i dataset).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MelSpectrogramExtractor(nn.Module):
    """
    Convert raw audio waveform to Mel spectrogram.

    Matches the preprocessing used in Fall-Mamba paper:
      - FFT → magnitude → Mel filterbank → log scale
      - Output: (B, 1, n_mels, time_frames) — 2D "image" representation
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalized: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.normalized = normalized

        # Mel filterbank (fixed)
        mel_fb = self._create_mel_filterbank()
        self.register_buffer('mel_filterbank', mel_fb)

        # Window function
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create Mel filterbank matrix."""
        n_freqs = self.n_fft // 2 + 1

        # Convert Hz to Mel scale
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Map Hz points to FFT bin indices
        freq_bins = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()
        freq_bins = torch.clamp(freq_bins, 0, n_freqs - 1)

        # Create filterbank
        filterbank = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            start, center, end = freq_bins[i], freq_bins[i + 1], freq_bins[i + 2]
            # Rising slope
            if center > start:
                filterbank[i, start:center] = (
                    torch.arange(0, center - start).float() / (center - start)
                )
            # Falling slope
            if end > center:
                filterbank[i, center:end] = 1.0 - (
                    torch.arange(0, end - center).float() / (end - center)
                )

        return filterbank

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, L) raw audio samples (1D or 2D)
        Returns:
            mel_spec: (B, 1, n_mels, time_frames) Mel spectrogram
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (B, n_fft//2+1, T)

        # Magnitude
        magnitude = torch.abs(stft)  # (B, n_fft//2+1, T)

        # Mel filterbank application
        mel = self.mel_filterbank @ magnitude  # (B, n_mels, T)

        # Log scale
        mel = torch.log(mel + 1e-10)

        # Normalize
        if self.normalized:
            mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / (
                mel.std(dim=(-2, -1), keepdim=True) + 1e-8
            )

        # Add channel dimension: (B, 1, n_mels, T)
        mel = mel.unsqueeze(1)

        return mel


class AudioChangeDetector(nn.Module):
    """
    Lightweight audio event detector for the 24fps pipeline.

    Detects sudden changes in audio (e.g., impact sound during a fall)
    and signals the JEPA inference thread to pay attention.
    """

    def __init__(self, energy_threshold: float = 2.0, history_size: int = 100):
        super().__init__()
        self.energy_threshold = energy_threshold
        self.history = []
        self.history_size = history_size
        self.baseline_energy = None

    def update(self, audio_chunk: torch.Tensor) -> bool:
        """
        Check if audio chunk contains an event.

        Args:
            audio_chunk: (L,) or (B, L) audio samples
        Returns:
            has_event: True if significant energy change detected
        """
        energy = float(torch.mean(audio_chunk ** 2))

        self.history.append(energy)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        if len(self.history) < 10:
            return False

        # Compute baseline from history
        baseline = sum(self.history) / len(self.history)

        # Detect deviation
        return energy > baseline * self.energy_threshold


class FrameMotionDetector(nn.Module):
    """
    Lightweight frame-to-frame motion detector.

    Detects significant visual changes between consecutive frames
    to trigger JEPA inference. Reduces unnecessary inference on static scenes.
    """

    def __init__(self, motion_threshold: float = 0.05):
        super().__init__()
        self.motion_threshold = motion_threshold
        self.prev_frame = None

    def update(self, frame: torch.Tensor) -> bool:
        """
        Check if current frame differs significantly from previous.

        Args:
            frame: (C, H, W) or (1, C, H, W) image tensor in [0, 1]
        Returns:
            has_motion: True if significant motion detected
        """
        if frame.dim() == 4:
            frame = frame.squeeze(0)

        if self.prev_frame is None:
            self.prev_frame = frame.clone()
            return False

        # Compute frame difference
        diff = torch.mean(torch.abs(frame - self.prev_frame))
        self.prev_frame = frame.clone()

        return diff > self.motion_threshold
