import torch
import matplotlib.pyplot as plt

def wave_to_spec(waveform, n_fft=2048, hop_length=512, win_length=2048):
    """
    Args:
        waveform: a torch tensor of one song in waveform
        n_fft: the number of frequency bins (determines resolution)
        hop_length: the step size between windows (determines time resolution)
        win_length: usually same as n_fft
    """
    window = torch.hann_window(win_length).to(waveform.device)
    spec = torch.stft(
            waveform,           # [C, T] or [T] if mono
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,  # if False, you'll get real+imag separately
            center=False
            )
    magnitude = spec.abs()

    return magnitude


def extract_waveform_segments(waveform, segment_samples, stride_samples):
    """
    waveform: [2, samples]
    Returns: [N, 2, segment_samples]
    """
    C, T = waveform.shape
    segments = []

    for start in range(0, T - segment_samples + 1, stride_samples):
        segment = waveform[:, start:start+segment_samples]  # [2, segment_samples]
        segments.append(segment)

    return torch.stack(segments, dim=0)  # [N, 2, segment_samples]


def extract_spec_patches(spec, patch_length, stride):
    """
    spec: [2, freq_bins, time_frames]
    Returns: [N, 2, freq_bins, patch_length]
    """
    C, F, T = spec.shape
    patches = []

    for t in range(0, T - patch_length + 1, stride):
        patch = spec[:, :, t:t+patch_length]  # [2, F, patch_length]
        patches.append(patch)

    return torch.stack(patches, dim=0)  # [N, 2, F, patch_length]

def visualize_wave_spec_pair(wave_segments, spec_patches, index=0, sample_rate=44100):
    """
    Visualizes one pair of wave segment and corresponding spectrogram patch.
    
    Args:
        wave_segments: Tensor [N, 2, segment_samples]
        spec_patches:  Tensor [N, 2, freq_bins, patch_length]
        index: which segment to show
        sample_rate: for time axis (in waveform)
    """
    wave = wave_segments[index]         # [2, segment_samples]
    spec = spec_patches[index]         # [2, freq_bins, patch_length]
    spec = torch.log10(spec + 1e-6)    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    time_axis = torch.arange(wave.shape[1]) / sample_rate

    # --- Waveform Plots ---
    axes[0, 0].plot(time_axis, wave[0].cpu().numpy(), label='Left')
    axes[0, 0].set_title("Waveform - Left Channel")
    axes[0, 1].plot(time_axis, wave[1].cpu().numpy(), label='Right', color='orange')
    axes[0, 1].set_title("Waveform - Right Channel")

    # --- Spectrogram Plots ---
    im0 = axes[1, 0].imshow(spec[0].cpu().numpy(), aspect='auto', origin='lower')
    axes[1, 0].set_title("Spectrogram Patch - Left")
    fig.colorbar(im0, ax=axes[1, 0])

    im1 = axes[1, 1].imshow(spec[1].cpu().numpy(), aspect='auto', origin='lower')
    axes[1, 1].set_title("Spectrogram Patch - Right")
    fig.colorbar(im1, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Constants
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    spec_patch_length = 5               # number of frames per spectrogram patch
    segment_samples = hop_length * spec_patch_length  # waveform samples per segment (aligned)
    stride_frames = 2                   # spectrogram patch stride
    stride_samples = hop_length * stride_frames  # waveform stride to match

    waveform = torch.rand((2, 100000))
    spec = wave_to_spec(waveform)
    
    wave_segments = extract_waveform_segments(waveform, segment_samples, stride_samples)
    spec_patches = extract_spec_patches(spec, patch_length=spec_patch_length, stride=stride_frames)

    print(wave_segments.shape, spec_patches.shape)