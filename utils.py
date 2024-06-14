import torch


def dynamic_collate_fn(batch: list, tokenizer_tgt) -> dict:
    """
    Dynamically pads the batch of data samples to the maximum sequence length.

    Args:
        batch (list): A list of dictionaries, where each dictionary represents a data sample.
                      Each dictionary should contain the following keys:
                      - "encoder_input": Tensor representing the encoder input sequence.
                      - "decoder_input": Tensor representing the decoder input sequence.
                      - "label": Tensor representing the label sequence.
        tokenizer_tgt: The target tokenizer used for padding.

    Returns:
        dict: A dictionary containing the padded batch of data samples with the following keys:
              - "encoder_input": Tensor representing the padded encoder input sequences.
              - "decoder_input": Tensor representing the padded decoder input sequences.
              - "encoder_mask": Tensor representing the encoder input mask.
              - "decoder_mask": Tensor representing the decoder input mask.
              - "label": Tensor representing the padded label sequences.
              - "src_text": List of source texts.
              - "tgt_text": List of target texts.
    """
    if not batch:
        raise ValueError("The input batch is empty.")

    # Find the maximum sequence lengths in the batch
    enc_len = max(len(item["encoder_input"]) for item in batch)
    dec_len = max(len(item["decoder_input"]) for item in batch)

    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

    encoder_input = []
    decoder_input = []
    label = []

    for item in batch:
        enc_item = item["encoder_input"]
        dec_item = item["decoder_input"]
        label_item = item["label"]

        # Pad the encoder input
        enc_item = torch.cat(
            [enc_item, torch.tensor((enc_len - len(enc_item),), pad_token_id, dtype=torch.int64)]
        )

        # Pad the decoder input
        dec_item = torch.cat(
            [dec_item, torch.tensor((dec_len - len(dec_item),), pad_token_id, dtype=torch.int64)]
        )

        # Pad the label
        label_item = torch.cat(
            [label_item, torch.tensor((dec_len - len(label_item),), pad_token_id, dtype=torch.int64)]
        )

        encoder_input.append(enc_item)
        decoder_input.append(dec_item)
        label.append(label_item)

    # Stack the padded sequences into tensors
    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    label = torch.stack(label)

    # Create masks for the encoder and decoder inputs
    encoder_mask = (encoder_input != pad_token_id).unsqueeze(1).unsqueeze(1).int()

    batch_size = decoder_input.size(0)
    causal_mask_ = causal_mask(dec_len).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    decoder_mask = (decoder_input != pad_token_id).unsqueeze(1).unsqueeze(2).int()
    decoder_mask = decoder_mask & causal_mask_.int()

    # Extract source and target texts
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    return {
        "encoder_input": encoder_input,  # (batch_size, seq_len)
        "decoder_input": decoder_input,  # (batch_size, seq_len)
        "encoder_mask": encoder_mask,  # (batch_size, 1, 1, seq_len)
        "decoder_mask": decoder_mask,  # (batch_size, 1, seq_len, seq_len)
        "label": label,  # (batch_size, seq_len)
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }


def causal_mask(size):
    """
    Generate a causal mask for self-attention mechanism.

    Args:
        size (int): The size of the mask.

    Returns:
        torch.Tensor: The causal mask with shape (1, size, size), where the upper triangle values are set to 0 and the lower triangle values are set to 1.

    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
