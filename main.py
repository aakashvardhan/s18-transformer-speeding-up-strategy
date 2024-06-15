import os
import warnings
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from config_file import get_config, get_weights_file_path
from dataset import LiTDataModule
from utils import get_model, greedy_decode


# Clear CUDA cache and set environment variable
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"

# Load configuration
config = get_config()


class LTModel(L.LightningModule):
    def __init__(self, cfg, tokenizer_src, tokenizer_tgt):
        super(LTModel, self).__init__()
        self.cfg = cfg
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.num_examples = cfg["num_examples"]
        self.initial_epoch = 0

        self.one_cycle_best_lr = cfg["one_cycle_best_lr"]
        self.learning_rate = config["lr"]

        self.model = get_model(
            config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
        )
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, eps=1e-9
        )
        self.writer = SummaryWriter(config["experiment_name"])

        self.source_texts = []
        self.expected = []
        self.predicted = []
        self.train_losses = []

        self.save_hyperparameters()

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        return self.model.project(decoder_output)

    def on_train_start(self):
        if config["preload"]:
            model_filename = get_weights_file_path(config, config["preload"])
            print("Preloading model {model_filename}")
            state = torch.load(model_filename)
            self.model.load_state_dict(state["model_state_dict"])
            self.initial_epoch = state["epoch"] + 1
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            print("preloaded")

    def training_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        label = batch["label"]

        proj_output = self(encoder_input, decoder_input, encoder_mask, decoder_mask)
        tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        loss = self.loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())
        self.writer.add_scalar("train_loss", loss.item(), self.trainer.global_step)
        self.writer.flush()
        return loss

    def validation_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]

        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

        model_out = greedy_decode(
            self.model,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.cfg["seq_len"],
            self.cfg["device"],
        )

        src_text = batch["src_text"][0]
        tgt_text = batch["tgt_text"][0]

        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        self.source_texts.append(src_text)
        self.expected.append(tgt_text)
        self.predicted.append(model_out_text)

    def on_validation_epoch_end(self):

        # Print 5 examples
        for _ in range(self.num_examples):
            idx = random.randint(0, len(self.source_texts) - 1)
            print("-" * 80)
            print(f"{f'SOURCE: ':>12}{self.source_texts[idx]}")
            print(f"{f'TARGET: ':>12}{self.expected[idx]}")
            print(f"{f'PREDICTED: ':>12}{self.predicted[idx]}")

        if self.writer:

            cer_metric = CharErrorRate()
            cer = cer_metric(self.predicted, self.expected)
            self.writer.add_scalar("validation cer", cer, self.trainer.global_step)
            self.writer.flush()

            wer_metric = WordErrorRate()
            wer = wer_metric(self.predicted, self.expected)
            self.writer.add_scalar("validation wer", wer, self.trainer.global_step)
            self.writer.flush()

            bleu_metric = BLEUScore()
            bleu = bleu_metric(self.predicted, self.expected)
            self.writer.add_scalar("validation BLEU", bleu, self.trainer.global_step)

            self.writer.flush()

    def on_save_checkpoint(self, checkpoint):
        model_filename = get_weights_file_path(self.cfg, f"{self.current_epoch}")
        print(f"Saving model to {model_filename}")
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.trainer.global_step,
            },
            model_filename,
        )

    def configure_optimizers(self):
        dataloader = self.trainer.datamodule.train_dataloader()
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.one_cycle_best_lr,
            steps_per_epoch=len(dataloader),
            epochs=self.trainer.max_epochs,
            pct_start=1 / 10 if self.config["num_epochs"] != 1 else 0.5,
            div_factor=10,
            three_phase=False,
            final_div_factor=10,
            anneal_strategy="linear",
        )
        return [self.optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss",
            }
        ]


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
    datamodule = LiTDataModule(cfg)
    datamodule.prepare_data()
    print("DataModule initialized...")
    tokenizer_src, tokenizer_tgt = datamodule.tokenizer_src, datamodule.tokenizer_tgt
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.getcwd(), version=1, name="lightning_logs"
    )

    if debug:
        trainer = L.Trainer(fast_dev_run=True, accelerator="cuda", devices="auto")
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


# if __name__ == "__main__":
#     main(config, debug=True)
