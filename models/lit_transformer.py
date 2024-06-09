import os

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from models.transformer import ProjectionLayer, Transformer, build_transformer
from utils import casual_mask

import random


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source_mask).fill_(next_word.item()),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


class LT_model(L.LightningModule):
    """
    LightningModule for the LT_model.

    Args:
        config (dict): Configuration parameters for the model.
        tokenizer_src (Tokenizer): Source tokenizer.
        tokenizer_tgt (Tokenizer): Target tokenizer.
    """

    def __init__(self, config, tokenizer_src, tokenizer_tgt, one_cycle_best_LR=0.01):
        super().__init__()
        self.config = config
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_vocab_size = self.tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        self.model = get_model(config, self.src_vocab_size, self.tgt_vocab_size)
        
        self.learning_rate = config["lr"]
        self.one_cycle_best_LR = one_cycle_best_LR

        self.source_texts = []
        self.expected = []
        self.predicted = []
        
        self.training_step_outputs = []

        self.cer_metric = CharErrorRate()
        self.wer_metric = WordErrorRate()
        self.bleu_metric = BLEUScore()

        self.save_hyperparameters()

    def forward(self, batch):
        """
        Forward pass of the LT_model.

        Args:
            encoder_input (Tensor): Input to the encoder.
            decoder_input (Tensor): Input to the decoder.
            encoder_mask (Tensor): Mask for the encoder input.
            decoder_mask (Tensor): Mask for the decoder input.

        Returns:
            Tensor: Projected output of the model.
        """

        encoder_input = batch["encoder_input"]  # (batch_size, seq_len)
        decoder_input = batch["decoder_input"]  # (batch_size, seq_len)

        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]

        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        proj_output = self.model.project(decoder_output)
        return proj_output

    def loss_fn(self, proj_output, label):
        # Define the loss function
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tk_tgt.token_to_id("[PAD]"), label_smoothing=0.1
        )
        loss = loss_fn(proj_output.view(-1, self.tgt_vocab_size), label.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step of the LT_model.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value.
        """
        proj_output = self(batch)
        label = batch["label"]  # (batch_size, seq_len)
        loss = self.loss_fn(proj_output, label)
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.training_step_outputs.append(loss)

        return loss

    def on_train_epoch_start(self):
        # Clear or reset the training_step_outputs at the beginning of each epoch
        self.training_step_outputs.clear()

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            epoch_mean_loss = torch.stack(self.training_step_outputs).mean()
            self.log("train_loss", epoch_mean_loss, on_epoch=True)
        else:
            print("Warning: No outputs to stack for epoch mean loss calculation.")

    def on_load_checkpoint(self, checkpoint):
        # Clear or restore self.training_step_outputs as needed
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer for the LT_model.

        Returns:
            Optimizer: The optimizer.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-9
        )

        dataloader = self.trainer.datamodule.train_dataloader()

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.one_cycle_best_LR,
            steps_per_epoch=len(dataloader),
            epochs=self.trainer.max_epochs,
            pct_start=0.2,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss",
        }

        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the LT_model.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing validation metrics.
        """

        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        model_out = greedy_decode(
            self.model,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["seq_len"],
        )
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)

    def on_validation_epoch_end(self):

        # Compute validation metrics
        cer = self.cer_metric(self.predicted, self.expected)
        wer = self.wer_metric(self.predicted, self.expected)
        bleu = self.bleu_metric(self.predicted, self.expected)

        for _ in range(self.config["num_examples"]):
            idx = random.randint(0, len(self.source_texts) - 1)
            print("-" * 80)
            print(f"{f'SOURCE: ':>12}{self.source_texts[idx]}")
            print(f"{f'TARGET: ':>12}{self.expected[idx]}")
            print(f"{f'PREDICTED: ':>12}{self.predicted[idx]}")

        # Log the validation loss dictionary
        self.log_dict(
            {"val_cer": cer, "val_wer": wer, "val_bleu": bleu},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
