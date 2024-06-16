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
    text = [item for item in text if len(item["translation"][src_config]) <= 150
            and len(item["translation"][tgt_config]) <= 150]

    text = [item for item in text if len(item["translation"][src_config]) + 10
            >= len(item["translation"][tgt_config])]
    return text


def dynamic_collate_fn(batch, tokenizer_tgt):
    
    from dataset import causal_mask
    
    # Dynamic batch padding
    # Find max seq_len in batch
    # max_len = max(list(map(lambda x: x["max_len"], batch)))
    enc_len = max([len(item['encoder_input']) for item in batch])
    dec_len = max([len(item['decoder_input']) for item in batch])

    encoder_input = []
    decoder_input = []
    label = []

    for item in batch:
        enc_item = item['encoder_input']
        dec_item = item['decoder_input']
        label_item = item['label']

        # Pad the encoder input
        enc_item = torch.cat(
            [
                enc_item,
                torch.tensor([tokenizer_tgt.token_to_id("[PAD]")] * 
                             (enc_len - len(enc_item)), dtype=torch.int64),
            ],
            dim=0
        )

        # Pad the decoder input
        dec_item = torch.cat(
            [
                dec_item,
                torch.tensor([tokenizer_tgt.token_to_id("[PAD]")] * 
                             (dec_len - len(dec_item)), dtype=torch.int64),
            ],
            dim=0
        )

        # Pad the label
        label_item = torch.cat(
            [
                label_item,
                torch.tensor([tokenizer_tgt.token_to_id("[PAD]")] * 
                             (dec_len - len(label_item)), dtype=torch.int64),
            ],
            dim=0
        )

        encoder_input.append(enc_item)
        decoder_input.append(dec_item)
        label.append(label_item)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    encoder_mask = (encoder_input != tokenizer_tgt.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).int()
    # Assume batch_size is the size of the batch, and dec_len is the length of the decoder sequences
    # Assume batch_size is the size of the batch, and dec_len is the length of the decoder sequences
    batch_size = decoder_input.size(0)

    # Generate the causal mask with the correct size
    causal_mask_ = causal_mask(dec_len).unsqueeze(1).repeat(batch_size, 1, 1,1)

    # Debugging: Print the shapes of the tensors
    # print("causal_mask new shape (after unsqueeze and repeat):", causal_mask.shape)
    decoder_mask = (decoder_input != tokenizer_tgt.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).int()
    # print("decoder_mask shape (after unsqueeze):", decoder_mask.shape)
    # print("causal_mask shape (after repeat):", causal_mask.shape)

    # The bitwise AND operation
    decoder_mask = decoder_mask & causal_mask_.int()

    
    label = torch.stack(label)
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]

    return {
        "encoder_input": encoder_input, # (batch_size, seq_len)
        "decoder_input": decoder_input, # (batch_size, seq_len)
        "encoder_mask": encoder_mask, # (batch_size, 1, 1, seq_len)
        "decoder_mask": decoder_mask, # (batch_size, 1, seq_len, seq_len)
        "label": label, # (batch_size, seq_len)
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }