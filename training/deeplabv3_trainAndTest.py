# General libraries
import os
# Utilities
import zipfile

# PyTorch and PyTorch Lightning
import torch.nn as nn
# Metrics
import torchmetrics
# Data Loaders
from data_loaders import SARDataModule
# Image processing
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
# Scikit-learn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# PyTorch Vision
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)
from utils import ZippingCheckpointCallback

# ============================= TRAINING MODULE =============================

class SARSegmentationModel(LightningModule):
    def __init__(self, learning_rate=5e-5, num_classes=5):
        super().__init__()
        self.save_hyperparameters()

        self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.train_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass")
        self.val_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass")

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.train_iou(preds, masks)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.val_iou(preds, masks)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.val_iou(preds, masks)  # You can use a separate metric if you prefer

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_iou", iou, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ============================= TRAINING LOOP =============================

def main():

    data_module = SARDataModule(data_dir="dataset/", batch_size=8, val_split=0.15)

    model = SARSegmentationModel(learning_rate=5e-5, num_classes=5)
    
    # Create callback to save best checkpoint during training
    checkpoint_callback = ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1, filename="best-checkpoint")

    zipping_callback = ZippingCheckpointCallback(checkpoint_callback=checkpoint_callback)

    # callbacks=checkpoint_callback
    # Pass both callbacks to the trainer
    trainer = Trainer(
        max_epochs=10,
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback, zipping_callback]
    )
    
    trainer.fit(model, datamodule=data_module)
    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = SARSegmentationModel.load_from_checkpoint(best_model_path)

# ============================= TEST =============================
    
    # Unzipping and loading the model for testing
    with zipfile.ZipFile(best_model_path, 'r') as zipf:
        zipf.extractall(os.path.dirname(best_model_path))
        unzipped_path = best_model_path.replace('.zip', '')

    best_model = SARSegmentationModel.load_from_checkpoint(unzipped_path)

    # Test the model
    trainer.test(best_model, datamodule=data_module)


if __name__ == '__main__':
    main()
