import os
import warnings
from pathlib import Path
import torch
import torchmetrics
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    TQDMProgressBar,
)


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config_file import get_config, get_weights_file_path
from dataset import BillingualDataset, LT_DataModule

from utils import dynamic_collate_fn, casual_mask

import torch.nn as nn
from models.lit_transformer import LT_model
from models.transformer import build_transformer


torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"
config = get_config()




def main(cfg, ckpt_file=None, if_ckpt=False):
    """
    Main function for training and evaluating a model.

    Args:
        cfg (dict): Configuration parameters for the model training.
        ckpt_file (str): Path to the checkpoint file.
        if_ckpt (bool, optional): Whether to load the model from a checkpoint. Defaults to False.
    """
    torch.cuda.empty_cache()
    L.seed_everything(42, workers=True)
    print("Seed set to 42...")

    # Initialize the data module
    datamodule = LT_DataModule(cfg)
    datamodule.setup()
    tk_src = datamodule.tokenizer_src  # Assuming datamodule is your instance of BilingualDataModule
    tk_tgt = datamodule.tokenizer_tgt

    src_vocab_size = tk_src.get_vocab_size()
    tgt_vocab_size = tk_tgt.get_vocab_size()

    print("DataModule initialized...")

    # Tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=1, name="lightning_logs"
    )

    # Initialize the trainer

    trainer = L.Trainer(
        precision=cfg["precision"],
        max_epochs=cfg["num_epochs"],
        logger=tb_logger,
        accelerator=cfg["accelerator"],
        devices="auto",
        default_root_dir=cfg["model_folder"],
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg["model_folder"],
                save_top_k=3,
                monitor="train_loss",
                mode="min",
                filename="model-{epoch:02d}-{train_loss:4f}",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
            EarlyStopping(monitor="train_loss", mode="min", stopping_threshold=1.7),
            TQDMProgressBar(refresh_rate=10),
        ],
        gradient_clip_val=0.5,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        enable_progress_bar=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        limit_val_batches=1000,
    )
    
    tuner = L.pytorch.tuner.Tuner(trainer)
    
    # Initialize the model
    model = LT_model(cfg, tk_src, tk_tgt)
    
    lr_finder = tuner.lr_find(
        model, datamodule=datamodule, num_training=trainer.max_epochs
    )
    print(lr_finder)
    
    # Check if the lr_finder has completed successfully
    if lr_finder:
        # Plot with suggest=True to find the suggested learning rate
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Get the suggested learning rate
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {suggested_lr}")
    else:
        print("Learning rate finding did not complete successfully.")

    # Train the model

    model.one_cycle_best_lr = suggested_lr

    if if_ckpt:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_file)
    else:
        trainer.fit(model=model, datamodule=datamodule)

    trainer.validate(model=model, datamodule=datamodule)
    print("Model Evaluation Done...")

    # Save the model
    torch.save(
        model.state_dict(),
        "saved_resnet18_model.pth",
    )
    print("Model saved...")
