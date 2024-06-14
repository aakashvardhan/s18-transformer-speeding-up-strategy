import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from utils import causal_mask, dynamic_collate_fn


class BillingualDataset(Dataset):
    """
    A PyTorch dataset for handling bilingual text data.

    Args:
        ds (Dataset): The original dataset containing the bilingual text data.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language.
        tgt_lang (str): The target language.
        seq_len (int): The maximum sequence length.

    Attributes:
        seq_len (int): The maximum sequence length.
        ds (Dataset): The original dataset containing the bilingual text data.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        src_lang (str): The source language.
        tgt_lang (str): The target language.
        sos_token (torch.Tensor): The tensor representing the start-of-sequence token.
        eos_token (torch.Tensor): The tensor representing the end-of-sequence token.
        pad_token (torch.Tensor): The tensor representing the padding token.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the encoder input, decoder input, encoder mask, decoder mask, label,
                  source text, and target text.

        Raises:
            ValueError: If the sentence is too long.

        """
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_str_length": len(enc_input_tokens),
            "decoder_str_length": len(dec_input_tokens),
        }


class LTDataModule(L.LightningDataModule):
    """
    LightningDataModule for handling data loading and processing in the LTDataModule class.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_all_sentenses(self, ds, lang):
        for item in ds:
            yield item["translation"][lang]

    def get_or_build_tokenizer(self, config, ds, lang):
        tokenizer_path = Path(config["tokenizer_file"].format(lang))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2
            )
            tokenizer.train_from_iterator(
                self.get_all_sentenses(ds, lang), trainer=trainer
            )
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer


    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            ds_raw = load_dataset(
            "opus_books", f"{self.config['lang_src']}-{self.config['lang_tgt']}", split="train"
            )

            src_lang = self.config["lang_src"]
            tgt_lang = self.config["lang_tgt"]
            seq_len = self.config["seq_len"]

            self.tokenizer_src = self.get_or_build_tokenizer(self.config, ds_raw, src_lang)
            self.tokenizer_tgt = self.get_or_build_tokenizer(self.config, ds_raw, tgt_lang)

            train_ds_size = int(0.9 * len(ds_raw))
            val_ds_size = len(ds_raw) - train_ds_size
            train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

            self.train_ds = BillingualDataset(
                train_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
            )
            self.val_ds = BillingualDataset(
                val_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
            )

            max_len_src = 0
            max_len_tgt = 0

            for item in ds_raw:
                src_ids = tokenizer_src.encode(item["translation"][src_lang]).ids
                tgt_ids = tokenizer_tgt.encode(item["translation"][tgt_lang]).ids
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))

            print(f"Max length of the source sentence : {max_len_src}")
            print(f"Max length of the source target : {max_len_tgt}")

    def train_dataloader(self):
        """
        This function is likely intended to create a data loader for training a machine learning model.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.config["n_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.config["n_workers"],
        )

    def collate_fn(self, batch):
        """
        Collates a batch of data samples.

        Args:
            batch (list): A list of data samples.

        Returns:
            torch.Tensor: The collated batch.

        """
        return dynamic_collate_fn(batch, self.tokenizer_tgt)

    def get_tokenizers(self):
        """
        Returns the source and target tokenizers used in the dataset.

        Returns:
            tokenizer_src (object): The tokenizer used for the source language.
            tokenizer_tgt (object): The tokenizer used for the target language.
        """
        return self.tokenizer_src, self.tokenizer_tgt
