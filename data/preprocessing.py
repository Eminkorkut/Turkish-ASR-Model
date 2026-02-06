"""
Audio preprocessing module using torchaudio for efficient feature extraction.
Supports Log-Mel Spectrograms with optional data augmentation.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from typing import Optional, Tuple

# Target sample rate for all audio
TARGET_SAMPLE_RATE = 16000


class AudioPreprocessor:
    """
    GPU-accelerated audio preprocessing using torchaudio.
    Replaces manual FFT-based feature extraction.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            sample_rate: Target sample rate (resamples if different)
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length between frames (10ms at 16kHz = 160)
            win_length: Window length (25ms at 16kHz = 400)
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank
            normalize: Whether to apply CMVN normalization
            device: Device for processing ("cpu" or "cuda")
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.normalize = normalize
        self.device = device
        
        # Mel Spectrogram Transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale="htk"
        ).to(device)
        
        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80).to(device)
        
    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample if necessary."""
        waveform, sr = torchaudio.load(path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        return waveform, self.sample_rate
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Log-Mel Spectrogram features from waveform.
        
        Args:
            waveform: (1, samples) or (samples,) tensor
            
        Returns:
            features: (Time, n_mels) tensor
        """
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        waveform = waveform.to(self.device)
        
        # Mel Spectrogram: (1, n_mels, time)
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale
        log_mel = self.amplitude_to_db(mel_spec)
        
        # Transpose to (time, n_mels) and remove batch dim
        features = log_mel.squeeze(0).transpose(0, 1)
        
        # Normalize (CMVN)
        if self.normalize:
            features = self._normalize(features)
            
        return features
    
    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Cepstral Mean and Variance Normalization (CMVN)."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)
    
    def __call__(self, path: str) -> torch.Tensor:
        """
        Full pipeline: load audio and extract features.
        
        Args:
            path: Path to audio file
            
        Returns:
            features: (Time, n_mels) tensor
        """
        waveform, _ = self.load_audio(path)
        return self.extract_features(waveform)


class SpecAugment(torch.nn.Module):
    """
    SpecAugment data augmentation for spectrograms.
    Applies frequency and time masking.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
        # Frequency masking
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        # Time masking
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment.
        
        Args:
            x: (batch, time, freq) or (time, freq) tensor
            
        Returns:
            Augmented tensor with same shape
        """
        # torchaudio expects (batch, freq, time) for masking
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim
            squeeze = True
        else:
            squeeze = False
            
        # Transpose to (batch, freq, time)
        x = x.transpose(1, 2)
        
        # Apply masks
        for _ in range(self.n_freq_masks):
            x = self.freq_masking(x)
        for _ in range(self.n_time_masks):
            x = self.time_masking(x)
            
        # Transpose back to (batch, time, freq)
        x = x.transpose(1, 2)
        
        if squeeze:
            x = x.squeeze(0)
            
        return x


class SpeedPerturbation:
    """
    Speed perturbation for audio data augmentation.
    Changes the speed of audio by a random factor.
    """
    
    def __init__(self, speeds: Tuple[float, ...] = (0.9, 1.0, 1.1)):
        self.speeds = speeds
        
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply random speed perturbation.
        
        Args:
            waveform: (1, samples) tensor
            sample_rate: Original sample rate
            
        Returns:
            Speed-perturbed waveform
        """
        speed = self.speeds[torch.randint(len(self.speeds), (1,)).item()]
        
        if speed == 1.0:
            return waveform
            
        if speed == 1.0:
            return waveform
            
        # Speed change via resampling (changes pitch and speed)
        # To speed up (speed > 1), we need fewer samples.
        # resample(orig, new) -> output_len = input_len * (new / orig)
        # We want output_len = input_len / speed
        # So new / orig = 1 / speed  => new = orig / speed
        new_freq = int(sample_rate / speed)
        
        perturbed = F.resample(waveform, orig_freq=sample_rate, new_freq=new_freq)
        
        return perturbed


class NoisePerturbation:
    """
    Add noise to audio for data augmentation.
    Supports SNR-based noise injection.
    """
    
    def __init__(
        self,
        noise_dir: Optional[str] = None,
        snr_range: Tuple[float, float] = (5.0, 20.0)
    ):
        """
        Args:
            noise_dir: Directory containing noise audio files
            snr_range: (min_snr, max_snr) in dB
        """
        self.noise_dir = noise_dir
        self.snr_range = snr_range
        self.noise_files = []
        
        if noise_dir:
            import glob
            self.noise_files = glob.glob(f"{noise_dir}/**/*.wav", recursive=True)
            
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Add noise at random SNR."""
        if not self.noise_files:
            return waveform
            
        # Load random noise file
        idx = torch.randint(len(self.noise_files), (1,)).item()
        noise, noise_sr = torchaudio.load(self.noise_files[idx])
        
        # Resample noise if needed
        if noise_sr != sample_rate:
            noise = T.Resample(noise_sr, sample_rate)(noise)
            
        # Match lengths
        if noise.shape[1] < waveform.shape[1]:
            # Repeat noise
            repeats = (waveform.shape[1] // noise.shape[1]) + 1
            noise = noise.repeat(1, repeats)
        noise = noise[:, :waveform.shape[1]]
        
        # Random SNR
        snr = torch.empty(1).uniform_(*self.snr_range).item()
        
        # Calculate scaling factor
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr / 10))))
        
        return waveform + scale * noise


# =====================================================
# Legacy API Compatibility
# =====================================================

# Create global preprocessor instance for backward compatibility
_preprocessor = None

def get_preprocessor(n_mel_channels: int = 80) -> AudioPreprocessor:
    """Get or create global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None or _preprocessor.n_mels != n_mel_channels:
        _preprocessor = AudioPreprocessor(n_mels=n_mel_channels)
    return _preprocessor


def pre_emphasis(signal, alpha=0.97):
    """Legacy: Pre-emphasis filter (now handled internally by torchaudio)."""
    import numpy as np
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    """Legacy: Framing function (now handled by MelSpectrogram)."""
    import numpy as np
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    return pad_signal[indices.astype(np.int32, copy=False)]


def windowing(frames):
    """Legacy: Hamming windowing (now handled by MelSpectrogram)."""
    import numpy as np
    frame_length = frames.shape[1]
    window = 0.54 - (0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1)))
    return frames * window


def power_spectrum(frames, nfft=512):
    """Legacy: Power spectrum (now handled by MelSpectrogram)."""
    import numpy as np
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    return (1.0 / nfft) * (mag_frames ** 2)


def get_filter_banks(pow_frames, sample_rate, nfilt=40, nfft=512):
    """Legacy: Mel filterbank (now handled by MelSpectrogram)."""
    import numpy as np
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return 20 * np.log10(filter_banks)


def normalize_features(features):
    """Legacy: CMVN normalization."""
    import numpy as np
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-8)
