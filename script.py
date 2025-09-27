import torch
import torch.nn as nn
from models import WaveEncoder, SpecEncoder, FusionNetwork, TransformerEncoder, MLPDecoder, SeparatorModel
from musedbdataset import SourceSeparationDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

from tqdm import tqdm


def collate_fn(batch):
    wave_list = [item['wave_segments'] for item in batch]
    spec_list = [item['spec_patches'] for item in batch]
    target_list = [item['target_vocals'] for item in batch]

    wave_lens = [seg.shape[0] for seg in wave_list]
    assert all(l == spec.shape[0] == tgt.shape[0]
               for l, spec, tgt in zip(wave_lens, spec_list, target_list)), "Mismatch in segment counts"

    # Pad along time (segment) axis
    wave_padded = pad_sequence(wave_list, batch_first=True)     # [B, max_N, 2, segment_samples]
    spec_padded = pad_sequence(spec_list, batch_first=True)     # [B, max_N, 2, F, patch_length]
    target_padded = pad_sequence(target_list, batch_first=True) # [B, max_N, 2, F, patch_length]

    # Segment mask
    max_len = max(wave_lens)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, l in enumerate(wave_lens):
        mask[i, :l] = 1

    return {
        'wave_segments': wave_padded,     # [B, N, 2, T]
        'spec_patches':  spec_padded,     # [B, N, 2, F, T_spec]
        'target_patches': target_padded,  # [B, N, 2, F, T_spec]
        'segment_mask':  mask,            # [B, N]
        'song_names':    [item['song_name'] for item in batch]
    }


def train_one_epoch(model, dataset, optimizer, criterion, device):
    model.train()
    total_loss = 0

    progress = tqdm(dataset, desc="Training", unit="song")

    for song in dataset:
        wave = song['wave_segments'].to(device)      # [N, 2, T]
        spec = song['spec_patches'].to(device)       # [N, 2, F, T_spec]
        target = song['target_vocals'].to(device)   # [N, 2, F, T_spec]

        optimizer.zero_grad()
        output = model(wave, spec)                   # [N, 2, F, T_spec]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataset)


@torch.no_grad()
def validate(model, dataset, criterion, device):
    model.eval()
    total_loss = 0

    progress = tqdm(dataset, desc="Validation", unit="song")

    for song in dataset:
        wave = song['wave_segments'].to(device)
        spec = song['spec_patches'].to(device)
        target = song['target_vocals'].to(device)

        output = model(wave, spec)
        loss = criterion(output, target)
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataset)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)



def main():
    path = r"F:\musdb18hq"
    wave_encoder = WaveEncoder()
    spec_encoder = SpecEncoder()
    fusion_network = FusionNetwork()
    transformer_encoder = TransformerEncoder()
    decoder = MLPDecoder()
    dataset = SourceSeparationDataset(path)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))

    batch_wave = batch["wave_segments"]
    batch_spec = batch["spec_patches"]

    B_w, N_w, C_w, T_w = batch_wave.shape
    B_s, N_s, C_s, F_s, T_s = batch_spec.shape


    #with torch.no_grad():
    gpu_data_wave = batch_wave.view(B_w * N_w, C_w, T_w).to("cuda")
    gpu_data_spec = batch_spec.view(B_s * N_s, C_s, F_s, T_s).to("cuda")
    gpu_encoder_wave = wave_encoder.to("cuda")
    gpu_encoder_spec = spec_encoder.to("cuda")
    gpu_fusion_network = fusion_network.to("cuda")
    gpu_transformer_encoder = transformer_encoder.to("cuda")
    gpu_decoder = decoder.to("cuda")

    wave_f_out = gpu_encoder_wave(gpu_data_wave).view(B_w, N_s, -1)
    spec_f_out = gpu_encoder_spec(gpu_data_spec).view(B_s, N_s, -1)

    fused_out = gpu_fusion_network(wave_f_out, spec_f_out)
    contextualised = gpu_transformer_encoder(fused_out)
    spec_out = gpu_decoder(contextualised)

    print(gpu_data_wave.shape)
    print(gpu_data_spec.shape)

    print(wave_f_out.shape)
    print(spec_f_out.shape)
    print(fused_out.shape)
    print(contextualised.shape)
    print(spec_out.shape)

if __name__ == "__main__":
    # main()
    path = r"F:\musdb18hq"
    dataset = SourceSeparationDataset(path, "train")
    train_dataset = Subset(dataset, list(range(0, 80)))
    val_dataset = Subset(dataset, list(range(80, len(dataset))))
    num_epochs = 10

    best_val_loss = float('inf')
    best_epoch = -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SeparatorModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_dataset, optimizer, criterion, device)
        val_loss = validate(model, val_dataset, criterion, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, "best_model.pt")
            print(f"Best model updated at epoch {epoch} (val_loss={val_loss:.4f})")