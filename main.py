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


class LTModel(L.LightningModule):
    def __init__(self, cfg, tokenizer_src, tokenizer_tgt, train_dataloader, learning_rate=0.0003, one_cycle_best_lr=0.0001):
        super(LTModel, self).__init__()
        self.cfg = cfg
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.num_examples = self.cfg["num_examples"]
        self.initial_epoch = 0

        self.one_cycle_best_lr = one_cycle_best_lr
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader

        self.model = get_model(
            self.cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
        )
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-9
        )
        self.writer = SummaryWriter(self.cfg["experiment_name"])

        self.source_texts = []
        self.expected = []
        self.predicted = []
        self.train_losses = []

        self.save_hyperparameters()

    def forward(self, batch):
        encoder_input = batch["encoder_input"]  # (batch_size, seq_len)
        decoder_input = batch["decoder_input"]  # (batch_size, seq_len)

        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(
            encoder_input, encoder_mask
        )  # (batch_size, seq_len, d_model)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )  # (batch_size, seq_len, d_model)
        proj_output = self.model.project(
            decoder_output
        )  # (batch_size, seq_len, vocab_size)
        return proj_output

    def on_train_start(self):
        if self.cfg["preload"]:
            model_filename = get_weights_file_path(self.cfg, self.cfg["preload"])
            print("Preloading model {model_filename}")
            state = torch.load(model_filename)
            self.model.load_state_dict(state["model_state_dict"])
            self.initial_epoch = state["epoch"] + 1
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            print("preloaded")

    def training_step(self, batch, batch_idx):
        proj_output = self(batch)
        label = batch["label"]  # (batch_size, seq_len)
        tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        loss = self.loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        # assert loss is not nan
        # assert not torch.isnan(loss).any(), "Loss is NaN"

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
        
    def on_validation_start(self):
        self.source_texts = []
        self.expected = []
        self.predicted = []

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

    def configure_optimizers(self):
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.one_cycle_best_lr,
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.trainer.max_epochs,
            pct_start=6/self.trainer.max_epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        return [self.optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
            }
        ]


# def main(cfg, ckpt_file=None, if_ckpt=False, debug=False):
#     """
#     Main function for training and evaluating a model.

#     Args:
#         cfg (dict): Configuration parameters for the model training.
#         ckpt_file (str): Path to the checkpoint file.
#         if_ckpt (bool, optional): Whether to load the model from a checkpoint. Defaults to False.
#     """
#     # Define the directory name
#     directory_name = "weights"

#     # Create the directory if it does not exist
#     if not os.path.exists(directory_name):
#         os.makedirs(directory_name)
#         print(f"Directory '{directory_name}' created!")
#     else:
#         print(f"Directory '{directory_name}' already exists.")

#     # Clear CUDA cache and set seed
#     torch.cuda.empty_cache()
#     L.seed_everything(42, workers=True)
#     print("Seed set to 42...")

#     # Initialize the data module
#     datamodule = LiTDataModule(cfg)
#     datamodule.setup()
#     print("DataModule initialized...")
#     tokenizer_src, tokenizer_tgt = datamodule.tokenizer_src, datamodule.tokenizer_tgt
#     # Initialize TensorBoard logger
#     tb_logger = TensorBoardLogger(
#         save_dir=os.getcwd(), version=1, name="lightning_logs"
#     )

#     if debug:
#         trainer = L.Trainer(fast_dev_run=True, accelerator="cuda", devices="auto")
#         # Initialize the model
#         model = LTModel(cfg, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)
#         trainer.fit(model=model, datamodule=datamodule)
#         print("Debugging Done...")
#     else:
#         # Initialize the trainer
#         trainer = L.Trainer(
#             precision=cfg["precision"],
#             max_epochs=cfg["num_epochs"],
#             logger=tb_logger,
#             accelerator=cfg["accelerator"],
#             devices="auto",
#             default_root_dir=cfg["model_folder"],
#             callbacks=[
#                 ModelCheckpoint(
#                     dirpath=cfg["model_folder"],
#                     save_top_k=3,
#                     monitor="train_loss_step",
#                     mode="min",
#                     filename="model-{epoch:02d}-{train_loss:.4f}",
#                     save_last=True,
#                 ),
#                 LearningRateMonitor(logging_interval="step", log_momentum=True),
#                 EarlyStopping(
#                     monitor="train_loss_step", mode="min", stopping_threshold=1.6
#                 ),
#                 TQDMProgressBar(refresh_rate=10),
#             ],
#             gradient_clip_val=0.5,
#             num_sanity_val_steps=5,
#             enable_progress_bar=True,
#             check_val_every_n_epoch=1,
#             limit_val_batches=2,
#         )

#         # Initialize the model
#         model = LTModel(cfg, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt)

#         # Learning rate finder
#         tuner = L.pytorch.tuner.Tuner(trainer)
#         lr_finder = tuner.lr_find(
#             model, datamodule=datamodule, num_training=trainer.max_epochs
#         )
#         print(lr_finder)

#         # Initialize suggested_lr with a default value
#         suggested_lr = cfg["one_cycle_best_lr"]

#         if lr_finder:
#             fig = lr_finder.plot(suggest=True)
#             fig.show()
#             suggested_lr = lr_finder.suggestion()
#             print(f"Suggested learning rate: {suggested_lr}")
#         else:
#             print("Learning rate finding did not complete successfully.")

#         # Set the best learning rate
#         model.one_cycle_best_lr = suggested_lr

#         # Train the model
#         if if_ckpt:
#             trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_file)
#         else:
#             trainer.fit(model=model, datamodule=datamodule)

#         # Validate the model
#         trainer.validate(model=model, datamodule=datamodule)
#         print("Model Evaluation Done...")

#         # Save the model
#         torch.save(model.state_dict(), "saved_resnet18_model.pth")
#         print("Model saved...")


# if __name__ == "__main__":
#     main(config, debug=True)
