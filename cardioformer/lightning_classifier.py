import pytorch_lightning as pl
import torch 
import torch.nn as nn
from torchmetrics import Accuracy, AUROC
from transformer import Cardioformer
import utils
class CardioformerLightning(pl.LightningModule):
    def __init__(self, model_hparams, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = Cardioformer(**model_hparams)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([utils.POS_WEIGHT]))
        
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        signals, labels = batch
        # Handle cases where the batch is empty after filtering
        if signals.numel() == 0:
            return None, None, None

        logits = self(signals)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        # Skip step if the batch was empty
        if loss is None:
            return None
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(preds, labels.int())
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        if loss is None:
            return None

        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(preds, labels.int())
        self.val_auroc(preds, labels)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(preds)
        self.validation_step_labels.append(labels)
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.validation_step_outputs:
            return
            
        all_preds = torch.cat(self.validation_step_outputs).squeeze().cpu().numpy()
        all_labels = torch.cat(self.validation_step_labels).squeeze().cpu().numpy()

        challenge_score = utils.compute_challenge_score(all_labels, all_preds)
        self.log('val_challenge_score', challenge_score, prog_bar=True)
        
        print(f"\nEpoch {self.current_epoch}: Validation Challenge Score = {challenge_score:.4f}")

        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, batch_idx)
        if loss is None:
            return None

        self.log('test_loss', loss)
        self.test_acc(preds, labels.int())
        self.test_auroc(preds, labels)
        self.log('test_acc', self.test_acc, on_epoch=True, prog_bar=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True, prog_bar=True)

        self.test_step_outputs.append(preds)
        self.test_step_labels.append(labels)

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("No test outputs were generated, skipping score calculation.")
            return
        all_preds = torch.cat(self.test_step_outputs).squeeze().cpu().numpy()
        all_labels = torch.cat(self.test_step_labels).squeeze().cpu().numpy()

        challenge_score = utils.compute_challenge_score(all_labels, all_preds)
        self.log('test_challenge_score', challenge_score, prog_bar=True)

        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'], 
                                      weight_decay=self.hparams.optimizer_hparams['weight_decay'])

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.optimizer_hparams.get('warmup_steps', 0)
        lr_end = self.hparams.optimizer_hparams.get('lr_end', 1e-7)

        if warmup_steps <= 0:
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=lr_end
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": main_scheduler, "interval": "step"},
            }

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )   
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=lr_end
        )

        sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "step",
            },
        }