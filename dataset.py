import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class BillingualDataset(Dataset):
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
        return len(self.ds)

    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        # For encoding, we PAD both SOS and EOS. For decoding, we only pad SOS.
        # THe model is required to predict EOS and stop on its own.

        # Make sure that padding is not negative (ie the sentance is too long)
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
            # encoder mask: (1, 1, seq_len) -> Has 1 when there is text and 0 when there is pad (no text)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(decoder_input.size(0)),
            # (1, seq_len) and (1, seq_len, seq_len)
            # Will get 0 for all pads. And 0 for earlier text.
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_str_length": len(enc_input_tokens),
            "decoder_str_length": len(dec_input_tokens),
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    # This will get the upper traingle values
    return mask == 0


class LiTDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        from utils import clean_long_text

        ds_raw = load_dataset(
            "opus_books",
            f"{self.config['lang_src']}-{self.config['lang_tgt']}",
            split="train",
        )

        src_lang = self.config["lang_src"]
        tgt_lang = self.config["lang_tgt"]
        seq_len = self.config["seq_len"]

        tokenizer_src = self.get_or_build_tokenizer(ds_raw, src_lang)
        tokenizer_tgt = self.get_or_build_tokenizer(ds_raw, tgt_lang)

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        ds_raw = clean_long_text(self.config, ds_raw)

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
            src_ids = self.tokenizer_src.encode(item["translation"][src_lang]).ids
            tgt_ids = self.tokenizer_tgt.encode(item["translation"][tgt_lang]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of the source sentence : {max_len_src}")
        print(f"Max length of the source target : {max_len_tgt}")

    def collate_fn(self, batch):

        encoder_input_max = max(x["encoder_str_length"] for x in batch)
        decoder_input_max = max(x["decoder_str_length"] for x in batch)

        encoder_inputs = []
        decoder_inputs = []
        encoder_mask = []
        decoder_mask = []
        label = []
        src_text = []
        tgt_text = []

        for b in batch:
            encoder_inputs.append(b["encoder_input"][:encoder_input_max])
            decoder_inputs.append(b["decoder_input"][:decoder_input_max])
            encoder_mask.append(
                (b["encoder_mask"][0, 0, :encoder_input_max])
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .int()
            )
            decoder_mask.append(
                (b["decoder_mask"][0, :decoder_input_max, :decoder_input_max])
                .unsqueeze(0)
                .unsqueeze(0)
            )
            label.append(b["label"][:decoder_input_max])
            src_text.append(b["src_text"])
            tgt_text.append(b["tgt_text"])

        return {
            "encoder_input": torch.vstack(encoder_inputs),
            "decoder_input": torch.vstack(decoder_inputs),
            "encoder_mask": torch.vstack(encoder_mask),
            "decoder_mask": torch.vstack(decoder_mask),
            "label": torch.vstack(label),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["n_workers"],
            # collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config["n_workers"],
            # collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def get_or_build_tokenizer(self, ds, lang):
        tokenizer_path = Path(self.config["tokenizer_file"].format(lang))
        if not tokenizer_path.exists():
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2
            )
            tokenizer.train_from_iterator(
                self.get_all_sentences(ds, lang), trainer=trainer
            )
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer

    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item["translation"][lang]
