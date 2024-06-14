import os
import lightning as L
import torch

from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from lightning.pytorch.loggers import TensorBoardLogger

from config_file import get_config
from dataset import LTDataModule
from models.lit_transformer import LTModel

# Clear CUDA cache and set environment variable
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"

# Load configuration
config = get_config()


def main(cfg, ckpt_file=None, if_ckpt=False, debug=False):
    """
    Main function for training and evaluating a model.

    Args:
        cfg (dict): Configuration parameters for the model training.
        ckpt_file (str): Path to the checkpoint file.
        if_ckpt (bool, optional): Whether to load the model from a checkpoint. Defaults to False.
    """
    # Clear CUDA cache and set seed
    torch.cuda.empty_cache()
    L.seed_everything(42, workers=True)
    print("Seed set to 42...")

    # Initialize the data module
    datamodule = LTDataModule(cfg)
    datamodule.setup()
    print("DataModule initialized...")
    tokenizer_src, tokenizer_tgt = datamodule.tokenizer_src, datamodule.tokenizer_tgt
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=1, name="lightning_logs"
    )

    if debug:
        trainer = L.Trainer(fast_dev_run=True)
        # Initialize the model
        model = LTModel(cfg, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)
        trainer.fit(model=model, datamodule=datamodule)
        print("Debugging Done...")
    else:
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
                    filename="model-{epoch:02d}-{train_loss:.4f}",
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
            check_val_every_n_epoch=9,
            limit_val_batches=1000,
        )

        # Initialize the model
        model = LTModel(cfg, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)

        # Learning rate finder
        tuner = L.pytorch.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(
            model, datamodule=datamodule, num_training=trainer.max_epochs
        )
        print(lr_finder)
        
        # Initialize suggested_lr with a default value
        suggested_lr = cfg["one_cycle_best_lr"]

        if lr_finder:
            fig = lr_finder.plot(suggest=True)
            fig.show()
            suggested_lr = lr_finder.suggestion()
            print(f"Suggested learning rate: {suggested_lr}")
        else:
            print("Learning rate finding did not complete successfully.")

        # Set the best learning rate
        model.one_cycle_best_lr = suggested_lr

        # Train the model
        if if_ckpt:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_file)
        else:
            trainer.fit(model=model, datamodule=datamodule)

        # Validate the model
        trainer.validate(model=model, datamodule=datamodule)
        print("Model Evaluation Done...")

        # Save the model
        torch.save(model.state_dict(), "saved_resnet18_model.pth")
        print("Model saved...")


if __name__ == "__main__":
    main(config,debug=True)
