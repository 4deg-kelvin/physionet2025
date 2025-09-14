import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchmetrics.classification import Accuracy
import math

import utils
from wavelnet import WaveletConv, create_daubechies_wavelet

# --- Building Blocks (Refactored) ---

class SE_Module(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.
    This module adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, in_channels, ratio=16, dim=2):
        super(SE_Module, self).__init__()
        self.dim = dim
        
        # Squeeze operation: Global Average Pooling
        if self.dim == 1:
            self.squeeze = nn.AdaptiveAvgPool1d(1)
        else:
            self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
            
        # Excitation operation: Two fully-connected layers
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // ratio, out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, channels, length] for 1D or [batch, channels, height, width] for 2D
        identity = x
        
        # Squeeze
        out = self.squeeze(x)
        # Flatten for the linear layers
        out = out.view(out.size(0), -1)
        
        # Excitation
        scale = self.excitation(out)
        
        # Reshape scale to broadcast over the input dimensions
        if self.dim == 1:
            scale = scale.view(scale.size(0), scale.size(1), 1)
        else:
            scale = scale.view(scale.size(0), scale.size(1), 1, 1)
            
        # Recalibrate the input feature maps
        return identity * scale.expand_as(identity)


class ResBlock1d(nn.Module):
    """
    A 1D Residual Block with two main convolutional paths and SE modules.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, downsample=None):
        super(ResBlock1d, self).__init__()
        
        # First convolutional path
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.se1 = SE_Module(out_channels, dim=1)
        
        self.downsample = downsample
        
        # Second convolutional path
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.se2 = SE_Module(out_channels, dim=1)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # First residual connection
        identity1 = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.se1(out)
        
        if self.downsample is not None:
            identity1 = self.downsample(x)
            
        out += identity1
        
        # Second residual connection
        identity2 = out
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        
        out = self.se2(out)
        
        out += identity2
        out = self.relu(out)
        
        return out


class ResBlock2d(nn.Module):
    """
    A 2D Residual Block with an optional 3-convolution path and an SE module.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, downsample=None, num_conv=1):
        super(ResBlock2d, self).__init__()
        self.num_conv = num_conv
        
        # Main convolutional path
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if self.num_conv == 3:
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False) # Stride is 1 here
            self.bn3 = nn.BatchNorm2d(num_features=out_channels)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False) # Stride is 1 here
            
        self.se = SE_Module(out_channels, dim=2)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        if self.num_conv == 3:
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            
            out = self.bn3(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv3(out)
            
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# --- Main Model (Refactored) ---

class SE_ECGNet(nn.Module):
    """
    The main ECGNet model architecture with multi-branch feature extraction.
    """
    def __init__(self, struct=[(1, 3), (1, 5), (1, 7)], num_classes=1, in_channels=12, info_features=0):
        super(SE_ECGNet, self).__init__()
        self.struct = struct
        self.info_features = info_features
        # Store in_channels to validate against data shape in forward pass
        self.in_channels = in_channels
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 50), stride=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # First block of 2D residual layers
        self.block1 = self._make_layer_2d(32, 32, (1, 15), (1, 2), 3, (0, 7))
        
        # Multi-branch blocks
        self.block2_list = nn.ModuleList()
        self.block3_list = nn.ModuleList()
        
        for i, (k1, k2) in enumerate(self.struct):
            # 2D residual blocks for each branch
            block2 = self._make_layer_2d(32, 32, (k1, k2), (1, 1), 4, (0, k2 // 2))
            self.block2_list.append(block2)
            
            # 1D residual blocks for each branch
            block3 = self._make_layer_1d(32 * self.in_channels, 256, k2, 2, 4, k2 // 2)
            self.block3_list.append(block3)
            
        # Final classification layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=256 * len(struct) + self.info_features, out_features=num_classes)

    def _make_layer_1d(self, in_channels, out_channels, kernel_size, stride, num_blocks, padding):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = [ResBlock1d(in_channels, out_channels, kernel_size, stride, padding, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock1d(out_channels, out_channels, kernel_size, 1, padding))
            
        return nn.Sequential(*layers)

    def _make_layer_2d(self, in_channels, out_channels, kernel_size, stride, num_blocks, padding):
        num_conv = 3 if num_blocks == 4 else 1
        downsample = None
        if stride[1] != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = [ResBlock2d(in_channels, out_channels, kernel_size, stride, padding, downsample, num_conv)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock2d(out_channels, out_channels, kernel_size, (1, 1), padding, num_conv=num_conv))
            
        return nn.Sequential(*layers)

    def extract_features(self, x, info=None):
        """
        Run the shared backbone up to (but not including) the final fc layer,
        returning the feature vector.
        """
        out = x.unsqueeze(1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.block1(out)

        branch_outputs = []
        for i in range(len(self.struct)):
            sep = self.block2_list[i](out)
            sep = sep.view(sep.size(0), -1, sep.size(3))
            sep = self.block3_list[i](sep)
            sep = self.avgpool(sep)
            sep = sep.view(sep.size(0), -1)
            branch_outputs.append(sep)

        features = torch.cat(branch_outputs, dim=1)
        if self.info_features > 0 and info is not None:
            features = torch.cat([features, info], dim=1)
        return features

    def forward(self, x, info=None):
        features = self.extract_features(x, info)
        return self.fc(features)


class WaveletSE_ECGNet(SE_ECGNet):
    """
    SE_ECGNet with a learnable WaveletConv applied to each lead as the first step.
    This version corrects the initialization to handle the transformed input shape.
    """
    def __init__(self, *args,
                 wavelet_name: str = 'db6',
                 wavelet_kernel_size: int = 129,
                 wavelet_out_channels: int = 4,
                 **kwargs):
        
        # --- FIX STARTS HERE ---
        # The key is to update the `in_channels` for the parent SE_ECGNet.
        # The parent model needs to know the number of effective channels *after*
        # the wavelet transformation.
        
        # Get the original number of leads from kwargs.
        original_in_channels = kwargs.get('in_channels', 12) # Default to 12 if not provided
        
        # Calculate the new effective number of channels for the backbone.
        new_in_channels = original_in_channels * wavelet_out_channels
        
        # Update kwargs with the new, correct in_channels value before calling super().__init__.
        kwargs['in_channels'] = new_in_channels
        
        # Now, initialize the parent class with the corrected number of channels.
        # The parent will now correctly size its layers, including the BatchNorm1d.
        super().__init__(*args, **kwargs)
        # --- FIX ENDS HERE ---

        # Build the wavelet layer. This part is independent of the fix and was correct.
        # It operates on each lead individually (in_channels=1).
        mother = create_daubechies_wavelet(name=wavelet_name, kernel_size=wavelet_kernel_size)
        self.wavelet_conv = WaveletConv(
            in_channels=1,
            out_channels=wavelet_out_channels,
            kernel_size=wavelet_kernel_size,
            mother_wavelet=mother
        )
        self.wavelet_relu = nn.ReLU(inplace=True)

    def extract_features(self, x, info=None):
        # x: [batch, original_leads, seq_len]
        b, L, T = x.shape
        
        # Apply wavelet convolution to each lead separately.
        # Reshape to [batch * leads, 1, seq_len] to process each lead as a single channel.
        xw = x.view(b * L, 1, T)
        xw = self.wavelet_relu(self.wavelet_conv(xw))
        _, Cw, Tw = xw.shape
        
        # Reshape back to [batch, new_channels, new_seq_len]
        # new_channels = original_leads * wavelet_out_channels
        x_transformed = xw.view(b, L * Cw, Tw)
        
        # Now call the parent's feature extractor on this transformed tensor.
        # The parent is now correctly configured to handle this larger channel dimension.
        return super().extract_features(x_transformed, info)
class Wavelet_Pooled_SE_ECGNet(SE_ECGNet):
    """
    SE_ECGNet with a learnable WaveletConv and a subsequent pooling layer
    to reduce sequence length and improve training speed.
    """
    def __init__(self, *args,
                 wavelet_name: str = 'db6',
                 wavelet_kernel_size: int = 129,
                 wavelet_out_channels: int = 32,
                 **kwargs):
        
        # The key is to update the `in_channels` for the parent SE_ECGNet.
        original_in_channels = kwargs.get('in_channels', 12)
        new_in_channels = original_in_channels * wavelet_out_channels
        kwargs['in_channels'] = new_in_channels
        
        # Initialize the parent class with the corrected number of channels.
        super().__init__(*args, **kwargs)

        # Build the wavelet layer.
        mother = create_daubechies_wavelet(name=wavelet_name, kernel_size=wavelet_kernel_size)
        self.wavelet_conv = WaveletConv(
            in_channels=1,
            out_channels=wavelet_out_channels,
            kernel_size=wavelet_kernel_size,
            mother_wavelet=mother
        )
        self.wavelet_relu = nn.ReLU(inplace=True)
        
        # --- 1. DEFINE THE POOLING LAYER HERE ---
        # This layer will reduce the sequence length by a factor of 4.
        self.post_wavelet_pool = nn.AvgPool1d(kernel_size=4, stride=4)


    def extract_features(self, x, info=None):
        # x: [batch, original_leads, seq_len]
        b, L, T = x.shape
        
        # Apply wavelet convolution to each lead separately.
        xw = x.view(b * L, 1, T)
        xw = self.wavelet_relu(self.wavelet_conv(xw))
        
        # --- 2. APPLY THE POOLING LAYER HERE ---
        # This is the optimal place to downsample, right after feature extraction.
        xw = self.post_wavelet_pool(xw)
        
        # Reshape back to [batch, new_channels, new_seq_len]
        _, Cw, Tw = xw.shape
        x_transformed = xw.view(b, L * Cw, Tw)
        
        # Now call the parent's feature extractor on this smaller, transformed tensor.
        return super().extract_features(x_transformed, info)


# --- PyTorch Lightning Module ---

class LitSE_ECGNet(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the SE_ECGNet model.
    """
    def __init__(self,
                 num_classes=1,
                 learning_rate=3.33e-5,
                 weight_decay=1e-4,
                 in_channels=12,
                 info_features=2,
                 struct=[(1, 3), (1, 5), (1, 7)],
                 warmup_steps=500,
                 total_steps=10000,
                 use_wavelet_cnn: bool = False,
                 wavelet_name: str = 'db6',
                 wavelet_kernel_size: int = 129,
                 wavelet_out_channels: int = 32,
                 no_pooling: bool = True,  # New parameter to control pooling
                 do_multitask: bool = False,
                 rbbb_loss_weight: float = 0.25,
                 d1avb_loss_weight: float = 0.25): # <-- REMOVED chagas_loss_weight from signature
        super().__init__()
        # For binary classification, num_classes should be 1.
        if num_classes != 1:
            raise ValueError("For binary classification, `num_classes` must be 1.")
        
        # --- ROBUST WEIGHTING LOGIC ---
        # This ensures weights are positive and sum to 1.0, preventing negative loss issues.
        if do_multitask:
            if rbbb_loss_weight < 0 or d1avb_loss_weight < 0:
                raise ValueError("Auxiliary loss weights must be non-negative.")
            if rbbb_loss_weight + d1avb_loss_weight >= 1.0:
                raise ValueError("The sum of auxiliary loss weights must be less than 1.0.")
            # Calculate the main task weight implicitly.
            self.chagas_loss_weight = 1.0 - rbbb_loss_weight - d1avb_loss_weight
        
        self.save_hyperparameters()
        
        # choose backbone
        if self.hparams.use_wavelet_cnn:
            if self.hparams.no_pooling:
                self.model = WaveletSE_ECGNet(
                    num_classes=self.hparams.num_classes,
                    in_channels=self.hparams.in_channels,
                    info_features=self.hparams.info_features,
                    struct=self.hparams.struct,
                    wavelet_name=self.hparams.wavelet_name,
                    wavelet_kernel_size=self.hparams.wavelet_kernel_size,
                    wavelet_out_channels=self.hparams.wavelet_out_channels
                )
            else:
                self.model = Wavelet_Pooled_SE_ECGNet(
                    num_classes=self.hparams.num_classes,
                    in_channels=self.hparams.in_channels,
                    info_features=self.hparams.info_features,
                    struct=self.hparams.struct,
                    wavelet_name=self.hparams.wavelet_name,
                    wavelet_kernel_size=self.hparams.wavelet_kernel_size,
                    wavelet_out_channels=self.hparams.wavelet_out_channels
                )

        else:
            self.model = SE_ECGNet(
                num_classes=self.hparams.num_classes,
                in_channels=self.hparams.in_channels,
                info_features=self.hparams.info_features,
                struct=self.hparams.struct
            )
        
        # Use BCEWithLogitsLoss for binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics for binary classification
        task = "binary"
        self.train_accuracy = Accuracy(task=task)
        self.val_accuracy = Accuracy(task=task)
        self.test_accuracy = Accuracy(task=task)
        
        self.train_auroc = AUROC(task=task)
        self.val_auroc = AUROC(task=task)
        self.test_auroc = AUROC(task=task)

        # For storing outputs for epoch-level metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if self.hparams.do_multitask:
            # assume `self.model.fc` is your primary head
            head_dim = self.model.fc.in_features
            self.rbbb_head = nn.Linear(head_dim, 1)
            self.d1avb_head = nn.Linear(head_dim, 1)


    def forward(self, x, info=None):
        if not self.hparams.do_multitask:
            # single‐task: delegate to the base model
            return self.model(x, info)

        # multi‐task: use raw features to drive all three heads
        features = self.model.extract_features(x, info)
        main_logits = self.model.fc(features)
        rbbb_logits = self.rbbb_head(features)
        d1avb_logits = self.d1avb_head(features)
        return main_logits, rbbb_logits, d1avb_logits


    def _common_step(self, batch, batch_idx):
        if batch is None: # Handle empty batches from collate_fn
            return None, None, None

        if self.hparams.do_multitask:
            signals, info, chagas_labels, rbbb_labels, d1avb_labels = batch
            chagas_logits, rbbb_logits, d1avb_logits = self(signals, info)
        else:
            signals, info, chagas_labels = batch
            chagas_logits = self(signals, info)

        if signals.numel() == 0:
            return None, None, None

        # Always calculate the primary Chagas loss
        chagas_logits = chagas_logits.squeeze(-1)
        chagas_labels = chagas_labels.float().squeeze(-1)
        supervised_chagas_loss = self.criterion(chagas_logits, chagas_labels)

        # In training and multitask, combine losses
        if self.training and self.hparams.do_multitask:
            bce_loss_fn = nn.BCEWithLogitsLoss()
            rbbb_loss = torch.tensor(0.0, device=self.device)
            d1avb_loss = torch.tensor(0.0, device=self.device)

            # Squeeze to handle potential extra dims and create boolean mask
            rbbb_mask = (rbbb_labels != -1).squeeze()
            rbbb_mask = (rbbb_labels != -1).squeeze()
            if rbbb_mask.any():
                # --- FIX IS HERE ---
                # Squeeze the labels *before* applying the mask to align shapes.
                rbbb_loss = bce_loss_fn(rbbb_logits.squeeze(-1)[rbbb_mask], rbbb_labels.squeeze(-1)[rbbb_mask].float())

            d1avb_mask = (d1avb_labels != -1).squeeze()
            if d1avb_mask.any():
                # --- AND HERE ---
                d1avb_loss = bce_loss_fn(d1avb_logits.squeeze(-1)[d1avb_mask], d1avb_labels.squeeze(-1)[d1avb_mask].float())

            # Use the robustly calculated weights
            total_loss = (self.chagas_loss_weight * supervised_chagas_loss +
                          self.hparams.rbbb_loss_weight * rbbb_loss +
                          self.hparams.d1avb_loss_weight * d1avb_loss)
            
            # Log all losses during training
            self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_chagas_loss', supervised_chagas_loss, on_step=True, on_epoch=True)
            if rbbb_mask.any(): self.log('train_rbbb_loss', rbbb_loss, on_step=True, on_epoch=True)
            if d1avb_mask.any(): self.log('train_1davb_loss', d1avb_loss, on_step=True, on_epoch=True)
        else:
            # For validation/testing, or single-task training, loss is just the supervised Chagas loss
            total_loss = supervised_chagas_loss

        return total_loss, chagas_logits, chagas_labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        if loss is None:
            return None
        # The logging is now handled inside _common_step for multitask training
        if not (self.training and self.hparams.do_multitask):
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        if loss is None:
            return
            
        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        
        self.val_accuracy.update(logits, y.long())
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        self.val_auroc.update(logits, y.long())
        self.log('val_auroc', self.val_auroc, prog_bar=False, on_step=False, on_epoch=True)
        
        self.validation_step_outputs.append({'logits': logits, 'labels': y})

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        all_probs = torch.sigmoid(all_logits)
        
        challenge_score = utils.compute_challenge_score(all_labels.cpu().long(), all_probs.cpu())
        self.log('val_challenge_score', challenge_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        if loss is None:
            return

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        self.test_accuracy.update(logits, y.long())
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True)
        
        self.test_auroc.update(logits, y.long())
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True)
        
        self.test_step_outputs.append({'logits': logits, 'labels': y})

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        all_probs = torch.sigmoid(all_logits)
        
        challenge_score = utils.compute_challenge_score(all_labels.cpu().long(), all_probs.cpu())
        self.log('test_challenge_score', challenge_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="SE-ECGNet Model")
    parser.add_argument('--generate-dummy-data', action='store_true', 
                        help="Generate dummy data for testing (only use for debugging)")
    args = parser.parse_args()
    
    if not args.generate_dummy_data:
        print("Error: This script will not run without actual data.")
        print("Use --generate-dummy-data flag only for testing purposes.")
        print("For actual training, use the main training pipeline.")
        exit(1)
    
    print("WARNING: Running with dummy data for testing purposes only!")
    print("This should not be used for actual model training.")
    
    # --- Example Usage for Binary Classification ---
    
    BATCH_SIZE, NUM_CLASSES, NUM_LEADS, SEQ_LENGTH, INFO_FEATURES = 4, 1, 12, 5000, 10

    lit_model = LitSE_ECGNet(
        num_classes=NUM_CLASSES, 
        learning_rate=1e-3, 
        weight_decay=1e-4,
        info_features=INFO_FEATURES
    )
    
    dummy_ecg = torch.randn(BATCH_SIZE, NUM_LEADS, SEQ_LENGTH)
    dummy_info = torch.randn(BATCH_SIZE, INFO_FEATURES)
    # For binary classification, labels are 0 or 1
    dummy_labels = torch.randint(0, 2, (BATCH_SIZE,))

    print("--- Forward Pass Test ---")
    lit_model.eval()
    with torch.no_grad():
        logits = lit_model(dummy_ecg, dummy_info)
    print(f"Input ECG shape: {dummy_ecg.shape}, Info shape: {dummy_info.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (BATCH_SIZE, 1)
    print("Forward pass successful!\n")

    # --- Trainer Test ---
    dummy_train_loader = [(torch.randn(BATCH_SIZE, NUM_LEADS, SEQ_LENGTH), torch.randn(BATCH_SIZE, INFO_FEATURES), torch.randint(0, 2, (BATCH_SIZE,))) for _ in range(10)]
    dummy_val_loader = [(torch.randn(BATCH_SIZE, NUM_LEADS, SEQ_LENGTH), torch.randn(BATCH_SIZE, INFO_FEATURES), torch.randint(0, 2, (BATCH_SIZE,))) for _ in range(5)]
    dummy_test_loader = [(torch.randn(BATCH_SIZE, NUM_LEADS, SEQ_LENGTH), torch.randn(BATCH_SIZE, INFO_FEATURES), torch.randint(0, 2, (BATCH_SIZE,))) for _ in range(5)]

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        gradient_clip_val=0.5
    )

    print("--- Starting Dummy Training ---")
    trainer.fit(lit_model, train_dataloaders=dummy_train_loader, val_dataloaders=dummy_val_loader)
    print("\n--- Starting Dummy Testing ---")
    trainer.test(lit_model, dataloaders=dummy_test_loader)
    print("\nDummy run complete.")
    trainer.fit(lit_model, train_dataloaders=dummy_train_loader, val_dataloaders=dummy_val_loader)
    print("\n--- Starting Dummy Testing ---")
    trainer.test(lit_model, dataloaders=dummy_test_loader)
    print("\nDummy run complete.")
