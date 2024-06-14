from typing import Any, Dict, List, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

from config_file import get_weights_file_path
from models.transformer import build_transformer
from utils import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
    """
    Greedy decoding for sequence generation.

    Args:
        model (nn.Module): The transformer model.
        source (Tensor): Source input tensor.
        source_mask (Tensor): Source mask tensor.
        tokenizer_src (Tokenizer): Source tokenizer.
        tokenizer_tgt (Tokenizer): Target tokenizer.
        max_len (int): Maximum length of the generated sequence.

    Returns:
        Tensor: Decoded sequence.
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask)
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
    """
    Build the transformer model.

    Args:
        config (dict): Configuration parameters.
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.

    Returns:
        nn.Module: Transformer model.
    """
    return build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )


class LTModel(L.LightningModule):
    """
    LightningModule for the LTModel.

    Args:
        config (dict): Configuration parameters for the model.
        one_cycle_best_lr (float): Best learning rate for one cycle learning rate scheduler.
    """

    PAD_TOKEN = "[PAD]"

    def __init__(
        self,
        config: Dict[str, Any],
        tokenizer_src,
        tokenizer_tgt,
        one_cycle_best_lr: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_vocab_size = self.tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        self.model = get_model(config, self.src_vocab_size, self.tgt_vocab_size)
        self.learning_rate = config.get("lr", 0.001)
        self.one_cycle_best_lr = one_cycle_best_lr
        self.initial_epoch = 0
        self.source_texts: List[str] = []
        self.expected: List[str] = []
        self.predicted: List[str] = []
        self.training_step_outputs: List[torch.Tensor] = []
        self.cer_metric = CharErrorRate()
        self.wer_metric = WordErrorRate()
        self.bleu_metric = BLEUScore()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LTModel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected output of the model.
        """
        return self.model(x)

    def loss_fn(self, proj_output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the model.

        Args:
            proj_output (torch.Tensor): Projected output from the model.
            label (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_tgt.token_to_id(self.PAD_TOKEN),
            label_smoothing=0.1,
        )
        return loss_fn(proj_output.view(-1, self.tgt_vocab_size), label.view(-1))

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for the LTModel.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        label = batch["label"]
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        proj_output = self.model.project(decoder_output)
        loss = self.loss_fn(proj_output, label)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.training_step_outputs.append(loss)
        return loss

    def on_train_start(self) -> None:
        """
        Actions to perform at the start of training.
        """
        if self.config.get("preload"):
            model_filename = get_weights_file_path(self.config, self.config["preload"])
            state = torch.load(model_filename)
            self.model.load_state_dict(state["model_state_dict"])
            self.initial_epoch = state["epoch"] + 1
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            print("Preloaded")

    def on_train_epoch_start(self) -> None:
        """
        Actions to perform at the start of each training epoch.
        """
        self.training_step_outputs.clear()

    def on_train_epoch_end(self) -> None:
        """
        Actions to perform at the end of each training epoch.
        """
        if self.training_step_outputs:
            epoch_mean_loss = torch.stack(self.training_step_outputs).mean()
            self.log("train_loss_epoch", epoch_mean_loss, on_epoch=True)
        else:
            print("Warning: No outputs to stack for epoch mean loss calculation.")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Actions to perform when loading a checkpoint.

        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        self.training_step_outputs.clear()

    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[Dict[str, Any]]]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple: List containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-9)
        dataloader = self.trainer.datamodule.train_dataloader()
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.one_cycle_best_lr,
            steps_per_epoch=len(dataloader),
            epochs=self.trainer.max_epochs,
            pct_start=0.2,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss",
            }
        ]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step for the LTModel.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.
        """
        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]
        model_out = greedy_decode(
            self.model,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["seq_len"],
        )
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        self.source_texts.append(batch["src_text"][0])
        self.expected.append(batch["tgt_text"][0])
        self.predicted.append(model_out_text)

        if batch_idx < self.config.get("num_examples", 5):
            print("-" * 80)
            print(
                f"{f'Validation on example {batch_idx} on epoch {self.current_epoch}':^80}"
            )
            print(f"{f'SOURCE: ':>12}{batch['src_text'][0]}")
            print(f"{f'TARGET: ':>12}{batch['tgt_text'][0]}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")

    def on_validation_epoch_end(self) -> None:
        """
        Actions to perform at the end of each validation epoch.
        """
        cer = self.cer_metric(self.predicted, self.expected)
        wer = self.wer_metric(self.predicted, self.expected)
        bleu = self.bleu_metric(self.predicted, self.expected)
        self.log_dict(
            {"val_cer": cer, "val_wer": wer, "val_bleu": bleu},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Clear lists for the next epoch
        self.source_texts.clear()
        self.expected.clear()
        self.predicted.clear()
