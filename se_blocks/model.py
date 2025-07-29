import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchmetrics.classification import Accuracy
import math

import utils
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

    def forward(self, x, info=None):
        # Assert that the input data's channel dimension matches the model's configuration
        assert x.shape[1] == self.in_channels, \
            f"Input tensor has {x.shape[1]} leads (channels), but the model was instantiated with in_channels={self.in_channels}. Please ensure they match."

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
            
        out = torch.cat(branch_outputs, dim=1)
        
        if self.info_features > 0 and info is not None:
            out = torch.cat([out, info], dim=1)
            
        out = self.fc(out)
        return out


# --- PyTorch Lightning Module ---

class LitSE_ECGNet(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the SE_ECGNet model.
    """
    def __init__(self, num_classes=1, learning_rate=3.33e-5, weight_decay=1e-4, in_channels=12, info_features=0, struct=[(1, 3), (1, 5), (1, 7)], warmup_steps=500, total_steps=10000):
        super().__init__()
        # For binary classification, num_classes should be 1.
        if num_classes != 1:
            raise ValueError("For binary classification, `num_classes` must be 1.")
        self.save_hyperparameters()
        
        # FIX: Pass the in_channels hyperparameter to the underlying model
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

    def forward(self, x, info=None):
        return self.model(x, info)

    def _common_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, info, y = batch
        else:
            x, y = batch
            info = None
        
        logits = self(x, info)
        # Reshape logits and labels for BCEWithLogitsLoss
        # logits: [batch, 1] -> [batch], y: [batch] -> [batch]
        logits = logits.squeeze(1)
        y = y.float() # Ensure labels are float
        y = y.squeeze(1) if y.dim() > 1 else y
        
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Pass integer labels to metrics
        self.train_accuracy(logits, y.long())
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        self.train_auroc(logits, y.long())
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_accuracy(logits, y.long())
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        
        self.val_auroc(logits, y.long())
        self.log('val_auroc', self.val_auroc, prog_bar=True)
        
        self.validation_step_outputs.append({'logits': logits, 'labels': y})

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Apply sigmoid to get probabilities for the challenge score
        all_probs = torch.sigmoid(all_logits)
        
        challenge_score = utils.compute_challenge_score(all_labels.cpu().long(), all_probs.cpu())
        self.log('val_challenge_score', challenge_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        
        self.test_accuracy(logits, y.long())
        self.log('test_acc', self.test_accuracy)
        
        self.test_auroc(logits, y.long())
        self.log('test_auroc', self.test_auroc)
        
        self.test_step_outputs.append({'logits': logits, 'labels': y})

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()

    def on_test_epoch_end(self):
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # Apply sigmoid to get probabilities for the challenge score
        all_probs = torch.sigmoid(all_logits)
        
        challenge_score = utils.compute_challenge_score(all_labels.cpu().long(), all_probs.cpu())
        self.log('test_challenge_score', challenge_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / float(max(1, self.hparams.total_steps - self.hparams.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == '__main__':
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
