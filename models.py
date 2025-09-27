import torch
import torch.nn as nn
import math


class WaveEncoder(nn.Module):
    def __init__(self, in_channels=2, embedding_dim=256):
        super(WaveEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4),  # [B, 32, T/2]
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),           # [B, 64, T/4]
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),          # [B, 128, T/8]
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),         # [B, 256, T/16]
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)                                          # [B, 256, 1]
        )
        
        self.project = nn.Linear(256, embedding_dim)                         # Final projection to fixed size

    def forward(self, x):
        """
        Args:
            x: [B, 2, segment_samples] (e.g., [B, 2, 2560])
        Returns:
            [B, embedding_dim]
        """
        features = self.encoder(x)             # [B, 256, 1]
        features = features.squeeze(-1)        # [B, 256]
        return self.project(features)          # [B, embedding_dim]


class SpecEncoder(nn.Module):
    def __init__(self, in_channels=2, embedding_dim=256):
        super(SpecEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, F, T]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, F/2, T/2]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, F/4, T/4]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, F/8, T/8]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]
        )

        self.project = nn.Linear(256, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: [B, 2, freq_bins, patch_length]
        Returns:
            [B, embedding_dim]
        """
        x = self.encoder(x)         # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 256]
        return self.project(x)      # [B, embedding_dim]
    

class FusionNetwork(nn.Module):
    def __init__(self, input_dim=256, fused_dim=256):
        super(FusionNetwork, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, fused_dim),
        )

    def forward(self, wave_emb, spec_emb):
        """
        Args:
            wave_emb: [N, D]
            spec_emb: [N, D]
        Returns:
            fused: [N, fused_dim]
        """
        x = torch.cat([wave_emb, spec_emb], dim=-1)  # [N, 2D]
        return self.fusion(x)                        # [N, fused_dim]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [N, D] or [B, N, D]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # allows [B, N, D]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        """
        x: [N, D] or [B, N, D]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, D]
            single_input = True
        else:
            single_input = False

        x = self.positional_encoding(x)      # Add [B, N, D] + PE
        x = self.transformer(x)              # [B, N, D]

        if single_input:
            return x.squeeze(0)              # [N, D]
        return x
    

class MLPDecoder(nn.Module):
    def __init__(self, input_dim=256, freq_bins=1025, patch_length=20):
        super(MLPDecoder, self).__init__()

        output_dim = 2 * freq_bins * patch_length  # 2 channels

        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim)
        )

        self.freq_bins = freq_bins
        self.patch_length = patch_length

    def forward(self, x):
        """
        Args:
            x: [N, D] or [B, N, D]
        Returns:
            [N, 2, F, T] or [B, N, 2, F, T]
        """
        is_batched = x.dim() == 3  # [B, N, D]

        if is_batched:
            B, N, D = x.shape
            x = x.view(B * N, D)

        out = self.model(x)  # [B * N, 2 * F * T]
        out = out.view(-1, 2, self.freq_bins, self.patch_length)

        if is_batched:
            out = out.view(B, N, 2, self.freq_bins, self.patch_length)

        return out
    
class SeparatorModel(nn.Module):
    def __init__(self,
                 segment_samples=2560,
                 n_fft=2048,
                 patch_length=20,
                 embedding_dim=256):
        super().__init__()

        self.freq_bins = n_fft // 2 + 1
        self.patch_length = patch_length

        self.wave_encoder = WaveEncoder(in_channels=2, embedding_dim=embedding_dim)
        self.spec_encoder = SpecEncoder(in_channels=2, embedding_dim=embedding_dim)
        self.fusion = FusionNetwork(input_dim=embedding_dim, fused_dim=embedding_dim)
        self.transformer = TransformerEncoder(d_model=embedding_dim)
        self.decoder = MLPDecoder(input_dim=embedding_dim,
                                  freq_bins=self.freq_bins,
                                  patch_length=patch_length)

    def forward(self, wave_segments, spec_patches):
        """
        wave_segments: [N, 2, segment_samples]
        spec_patches: [N, 2, freq_bins, patch_length]
        Returns: [N, 2, freq_bins, patch_length]
        """
        N = wave_segments.shape[0]
        wave_emb = self.wave_encoder(wave_segments)         # [N, D]
        spec_emb = self.spec_encoder(spec_patches)          # [N, D]
        fused = self.fusion(wave_emb, spec_emb)             # [N, D]
        contextual = self.transformer(fused.unsqueeze(0))   # [1, N, D]
        decoded = self.decoder(contextual.squeeze(0))       # [N, 2, F, T]
        return decoded