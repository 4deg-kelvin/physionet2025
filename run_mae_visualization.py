#!/usr/bin/env python3
"""
Script to visualize masked and reconstructed ECG from a pretrained MAE.
"""
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt

# allow imports of project modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from mae.mae import MAELightningModule
from mae.dataloader import ECGDataModule


def unpatch(patched):
    """
    Convert patch representations back into time-series shape.
    patched: Tensor of shape (num_patches, patch_dim)
    Returns Tensor of shape (num_leads, seq_length)
    """
    P, D = patched.shape
    # assume D == num_leads * patch_size
    num_leads = 12
    patch_size = D // num_leads
    # reshape to (P, num_leads, patch_size)
    arr = patched.view(P, num_leads, patch_size)
    # reorder to (num_leads, P, patch_size)
    arr = arr.permute(1, 0, 2)
    # flatten to (num_leads, P * patch_size)
    return arr.reshape(num_leads, P * patch_size)


def main():
    parser = argparse.ArgumentParser(description="Visualize MAE reconstruction on ECG signals")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument('--batch_idx', type=int, default=0, help="Batch index in validation set to visualize")
    parser.add_argument('--lead', type=int, default=0, help="Which ECG lead to plot (0-11)")
    parser.add_argument('--loop', action='store_true', help="If set, iterate through all validation samples one at a time")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained Lightning module
    module = MAELightningModule.load_from_checkpoint(args.ckpt, map_location=device)
    module.eval()
    model = module.model.to(device)

    # Derive data params
    patch_size = module.hparams.patch_size
    num_patches = module.hparams.num_patches
    seq_len = patch_size * num_patches
    batch_size = 1

    # Prepare data module
    data_module = ECGDataModule(
        data_dir=os.path.join(project_root, 'training_data'),
        batch_size=batch_size,
        seq_len=seq_len,
        windowing_method='entire_recording'
    )
    data_module.setup()
    val_loader = data_module.val_dataloader()

    loader = val_loader
    def get_sample(idx):
        batch = next(iter(loader)) if not args.loop else None
        # if looping, idx indexes the iteration
        if args.loop:
            batch = list(loader)[idx]
        return batch[0][idx:idx+1].to(device)

    def process_and_plot(signals, idx):
        with torch.no_grad():
            recon_patches, orig_patches, masked_indices = model(signals)
        recon_patches = recon_patches[0].cpu()
        orig_patches = orig_patches[0].cpu()
        masked = masked_indices[0].cpu().tolist()
        orig_ts = unpatch(orig_patches)
        recon_ts = unpatch(recon_patches)
        lead = args.lead
        time = list(range(orig_ts.shape[1]))
        plt.figure(figsize=(12, 4))
        plt.plot(time, orig_ts[lead], label='Original', color='tab:blue')
        plt.plot(time, recon_ts[lead], label='Reconstructed', color='tab:orange', alpha=0.7)
        for m in masked:
            start = m * patch_size
            plt.axvspan(start, start + patch_size, color='gray', alpha=0.3)
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Voltage')
        plt.title(f'ECG Lead {lead}: Masked (shaded) and Reconstruction (Sample {idx})')
        plt.tight_layout()
        plt.show()

    if args.loop:
        total = len(list(val_loader))
        for idx in range(total):
            signals = get_sample(idx)
            process_and_plot(signals, idx)
            input("Press Enter to continue to next sample...")
    else:
        signals = get_sample(args.batch_idx)
        process_and_plot(signals, args.batch_idx)


if __name__ == '__main__':
    main()
