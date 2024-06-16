# This file contains the utility functions for the project
import torch
from model import build_transformer


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    from dataset import causal_mask

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(source_mask)
                .fill_(next_word.item())
                .to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def clean_long_text(config, text):
    src_config = config["lang_src"]
    tgt_config = config["lang_tgt"]
    text = sorted(text, key=lambda x: len(x["translation"][src_config]))
    text = [
        item
        for item in text
        if len(item["translation"][src_config]) <= 150
        and len(item["translation"][tgt_config]) <= 150
    ]

    text = [
        item
        for item in text
        if len(item["translation"][src_config]) + 10
        >= len(item["translation"][tgt_config])
    ]
    return text


def dynamic_collate_fn(batch):
    encoder_input_max = max(x["enc_token_len"] for x in batch)
    decoder_input_max = max(x["dec_token_len"] for x in batch)

    max_token_len = max([encoder_input_max, decoder_input_max])
    max_token_len_2 = max_token_len + 2

    encoder_inputs = []
    decoder_inputs = []
    encoder_masks = []
    decoder_masks = []
    labels = []
    src_texts = []
    tgt_texts = []

    for b in batch:
        encoder_inputs.append(b["encoder_input"][:max_token_len_2])
        decoder_inputs.append(b["decoder_input"][:max_token_len_2])
        encoder_mask = (
            (b["encoder_mask"][0, 0, :max_token_len_2])
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
        )
        encoder_masks.append(encoder_mask)
        decoder_mask = (
            (b["decoder_mask"][0, :max_token_len_2, :max_token_len_2])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        decoder_masks.append(decoder_mask)
        labels.append(b["label"][:max_token_len_2])
        src_texts.append(b["src_text"])
        tgt_texts.append(b["tgt_text"])

    return {
        "encoder_input": torch.vstack(encoder_inputs),
        "decoder_input": torch.vstack(decoder_inputs),
        "encoder_mask": torch.vstack(encoder_masks),
        "decoder_mask": torch.vstack(decoder_masks),
        "label": torch.vstack(labels),
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }
