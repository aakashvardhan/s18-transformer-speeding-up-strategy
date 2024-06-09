import os

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from transformer import ProjectionLayer, Transformer, build_transformer
from dataset import casual_mask

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

    def __init__(
        self, config, tokenizer_src, tokenizer_tgt
    ):
        super().__init__()
        self.config = config
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_vocab_size = self.tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()
        self.model = get_model(config, self.src_vocab_size, self.tgt_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], eps=1e-9
        )
        self.cer_metric = torchmetrics.CharErrorRate()
        self.wer_metric = torchmetrics.WordErrorRate()
        self.bleu_metric = torchmetrics.BLEUScore()

        self.save_hyperparameters()

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
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
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        proj_output = self.model.project(decoder_output)
        return proj_output

    def training_step(self, batch, batch_idx):
        """
        Training step of the LT_model.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value.
        """
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]
        label = batch["label"]

        proj_output = self(encoder_input, decoder_input, encoder_mask, decoder_mask)
        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the LT_model.

        Returns:
            Optimizer: The optimizer.
        """
        return self.optimizer

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the LT_model.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing validation metrics.
        """
        source_texts = []
        expected = []
        predicted = []

        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]
        label = batch["label"]
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        model_out = greedy_decode(
            self.model,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["seq_len"]
        )
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]

        source_texts.append(source_text)
        expected.append(target_text)
        predicted.append(model_out_text)

        # Compute validation metrics
        cer = self.cer_metric(predicted, expected)
        wer = self.wer_metric(predicted, expected)
        bleu = self.bleu_metric(predicted, expected)

        print("SOURCE: ", source_texts)
        print("-" * 80)
        print("TARGET: ", expected)
        print("-" * 80)
        print("PREDICTED: ", predicted)
        print("-" * 80)

        self.log("val_cer", cer, prog_bar=True)
        self.log("val_wer", wer, prog_bar=True)
        self.log("val_bleu", bleu, prog_bar=True)

        return {"val_cer": cer, "val_wer": wer, "val_bleu": bleu}
