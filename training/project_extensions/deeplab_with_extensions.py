# General libraries
import os
import sys
# Get the directory containing the current file and add it to Python's search path
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

# Utilities
import zipfile

# PyTorch and PyTorch Lightning
import torch.nn as nn
# Metrics
import torchmetrics
# Data Loaders
from extensions_data_module import SARDataModule
# Image processing
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
# Scikit-learn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# PyTorch Vision
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import ZippingCheckpointCallback




# ============================= TRAINING MODULE =============================

class SARSegmentationModel(LightningModule):
    def __init__(self, learning_rate=5e-5, num_classes=5):
        super().__init__()
        self.save_hyperparameters()

        self.model = deeplabv3_mobilenet_v3_large(weights=None)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.criterion = nn.CrossEntropyLoss(weights= [1.135639, 99.380251, 17.762885, 2347.162359, 18.992207])
        self.learning_rate = learning_rate

        self.train_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average='none')
        self.val_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average='none')

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.train_iou(preds, masks)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for class_idx, class_iou in enumerate(iou):
            self.log(f"train_iou_class_{class_idx}", class_iou, on_epoch=True, prog_bar=False)
        mean_iou = iou.mean()
        self.log("train_mean_iou", mean_iou, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.val_iou(preds, masks)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        for class_idx, class_iou in enumerate(iou):
            self.log(f"val_iou_class_{class_idx}", class_iou, on_epoch=True, prog_bar=False)
        mean_iou = iou.mean()
        self.log("val_mean_iou", mean_iou, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        preds = outputs.argmax(dim=1)
        iou = self.val_iou(preds, masks)  # You can use a separate metric if you prefer

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        for class_idx, class_iou in enumerate(iou):
            self.log(f"test_iou_class_{class_idx}", class_iou, on_epoch=True, prog_bar=False)
        mean_iou = iou.mean()
        self.log("test_mean_iou", mean_iou, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ============================= TRAINING LOOP =============================


def main():

    data_module = SARDataModule(data_dir="dataset/", batch_size=32, val_split=0.15)

    model = SARSegmentationModel(learning_rate=1e-3, num_classes=5)
    
    # Create callback to save best checkpoint during training
    checkpoint_callback = ModelCheckpoint(monitor="val_iou_class_1", mode="max", save_top_k=1, filename="best-checkpoint")

    zipping_callback = ZippingCheckpointCallback(checkpoint_callback=checkpoint_callback)

    # callbacks=checkpoint_callback
    # Pass both callbacks to the trainer
    trainer = Trainer(
        max_epochs=50,
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback, zipping_callback]
    )
    
    trainer.fit(model, datamodule=data_module)
    # Save the path of the best model
    best_model_path = checkpoint_callback.best_model_path
        
# ============================= TEST =============================

    with zipfile.ZipFile(best_model_path, 'r') as zipf:
        zipf.extractall(os.path.dirname(best_model_path))
        unzipped_path = best_model_path.replace('.zip', '')

    best_model = SARSegmentationModel.load_from_checkpoint(unzipped_path)
    
    # Test the model
    trainer.test(best_model, datamodule=data_module)


if __name__ == '__main__':
    main()
