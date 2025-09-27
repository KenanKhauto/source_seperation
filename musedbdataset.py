import os
import torch
from torch.utils.data import Dataset
import torchaudio
from utils import wave_to_spec, extract_waveform_segments, extract_spec_patches

class SourceSeparationDataset(Dataset):
    def __init__(self, root_dir, subset='train', sample_rate=44100, sources=None,
                 patch_length=20, stride_frames=15, hop_length=512, n_fft=2048):
        self.sample_rate = sample_rate
        self.sources = sources or ['bass', 'drums', 'other', 'vocals']
        self.subset_dir = os.path.join(root_dir, subset)
        self.song_dirs = [os.path.join(self.subset_dir, d) for d in os.listdir(self.subset_dir)
                          if os.path.isdir(os.path.join(self.subset_dir, d))]
        
        # Spectrogram paramters
        self.n_fft = n_fft
        self.hop_length = hop_length  # samples between STFT fames
        self.patch_length = patch_length  # How long each segment is (Number of spectrogram frames) -- how much audio context in each sample
        self.stride_frames = stride_frames  # How far you move forward to get the next segment (step size) -- How much overlap

        # Derived waveform slicing
        self.segment_samples = self.hop_length * self.patch_length
        self.stride_samples = self.hop_length * self.stride_frames
        

    def __len__(self):
        return len(self.song_dirs)

    def __getitem__(self, idx):
        song_path = self.song_dirs[idx]

        # Load mixture
        mixture_path = os.path.join(song_path, 'mixture.wav')
        mixture_waveform, sr = torchaudio.load(mixture_path)  # [2, T]
        if sr != self.sample_rate:
            mixture_waveform = torchaudio.functional.resample(mixture_waveform, sr, self.sample_rate)

        # Load vocals
        vocals_path = os.path.join(song_path, 'vocals.wav')
        vocals_waveform, sr_v = torchaudio.load(vocals_path)
        if sr_v != self.sample_rate:
            vocals_waveform = torchaudio.functional.resample(vocals_waveform, sr_v, self.sample_rate)

        # Get instrumental as target
        instrumental_waveform = mixture_waveform - vocals_waveform

        # Compute magnitude spectrograms
        spec_mix = wave_to_spec(mixture_waveform, self.n_fft, self.hop_length)
        spec_instr = wave_to_spec(instrumental_waveform, self.n_fft, self.hop_length)

        # Extract segments
        wave_segments = extract_waveform_segments(mixture_waveform, self.segment_samples, self.stride_samples)
        spec_patches = extract_spec_patches(spec_mix, self.patch_length, self.stride_frames)
        target_patches = extract_spec_patches(spec_instr, self.patch_length, self.stride_frames)

        # Align
        min_len = min(wave_segments.shape[0], spec_patches.shape[0], target_patches.shape[0])
        wave_segments = wave_segments[:min_len]
        spec_patches = spec_patches[:min_len]
        target_patches = target_patches[:min_len]

        assert wave_segments.shape[0] == spec_patches.shape[0] == target_patches.shape[0], "Mismatch in segment counts"

        return {
            'wave_segments': wave_segments,           # [N, 2, segment_samples]
            'spec_patches': spec_patches,             # [N, 2, F, patch_length]
            'target_vocals': target_patches,          # [N, 2, F, patch_length]
            'song_name': os.path.basename(song_path)
        }


if __name__ == "__main__":
    path = r"F:\musdb18hq"
    dataset = SourceSeparationDataset(path, "train")
    print(dataset[0])