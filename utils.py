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


def clean_long_text(config,text):
    src_config = config["lang_src"]
    tgt_config = config["lang_tgt"]
    text = sorted(
        text, key=lambda x: len(x["translation"][src_config])
    )
    text = [item for item in text if len(item["translation"][src_config]) <= 160
            and len(item["translation"][tgt_config]) <= 160]

    text = [item for item in text if len(item["translation"][src_config]) + 10
            >= len(item["translation"][tgt_config])]
    return text