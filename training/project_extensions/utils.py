import os
import zipfile

import numpy as np
import torch

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


class ZippingCheckpointCallback(Callback):
    def __init__(self, checkpoint_callback: ModelCheckpoint):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_path = self.checkpoint_callback.best_model_path

        if os.path.exists(checkpoint_path):
            # Define the zip filename
            zip_path = checkpoint_path + ".zip"
            
            # Compress the checkpoint
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(checkpoint_path, arcname=os.path.basename(checkpoint_path))
            
            # Optionally delete the original checkpoint to save space
            os.remove(checkpoint_path)
            print(f"Checkpoint compressed and saved at {zip_path}")