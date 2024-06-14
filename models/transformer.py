import torch
import torch.nn as nn
import math 
import lightning as pl
from lightning import LightningModule
# Layer Normalization: This is useful to keep bias false compared to built in torch layer norm
class LayerNormalization(LightningModule):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a trainable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a trainable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(-1, keepdim=True) # (batch_size, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(-1, keepdim=True) # (batch_size, seq_len, 1)
        # eps is a constant that prevents division by zero
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
# Feed Forward Layer
class FeedForwardBlock(LightningModule):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

# Input Embedding Layer
class InputEmbeddings(LightningModule):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) -> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to prevent the output of embedding layer to be too large, achieves numerical stability
        return self.embedding(x) * math.sqrt(self.d_model)
    

# Positional Encoding Layer
class PositionalEncoding(LightningModule):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create constant 'pe' matrix with values dependant on pos and i, shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000^(2i/d_model)))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000^(2i/d_model)))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the buffer to not be considered a model parameter, but position encoding is still a part of the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

# Residual Connection Layer
class ResidualConnection(LightningModule):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: LightningModule) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # sublayer: sublayer block
        return x + self.dropout(sublayer(self.norm(x)))
    

# Multi-Head Attention Layer
class MultiHeadAttentionBlock(LightningModule):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # d_model must be divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h # Dimension of each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # w_q
        self.w_k = nn.Linear(d_model, d_model, bias=False) # w_k
        self.w_v = nn.Linear(d_model, d_model, bias=False) # w_v
        self.w_o = nn.Linear(d_model, d_model, bias=False) # w_o
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query: torch.Tensor, 
                  key: torch.Tensor, 
                  value: torch.Tensor, 
                  mask: torch.Tensor, 
                  dropout: nn.Dropout):
        d_k = query.size(-1)
        # (batch, h, seq_len, d_k) x (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask is 0
            attention_scores.masked_fill_(mask == 0, -1e4)
            # print(attention_scores.shape)
        attention_scores = torch.softmax(attention_scores,dim=-1) # (batch, h, seq_len, seq_len) # softmax over the last dimension

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (batch, h, seq_len, seq_len) x (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization, attention score is taken seperately because we want to draw them in the future
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # b, sl, dm >> b, sl, h, dm/h >> b, h, sl, dm/h
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate Attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Combine heads
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # batch, h, seq_len, d_k << transpose
        # batch, seq_len, h, d_k << view
        # batch, seq_len, h * d_k << contiguous
        # Multiply by W_o
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)
    

class EncoderBlock(LightningModule):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, 
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        return self.residual_connections[1](x, self.feed_forward_block)
    

class Encoder(LightningModule):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(LightningModule):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # decoder input is passed to self attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(LightningModule):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(LightningModule):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(LightningModule):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask): # eg: BERT is a transformer encoder
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
    # eg:  BART: Encoder-Decoder Transformer


def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int=512, 
                      N: int=6, 
                      h: int=8, 
                      dropout: float=0.1, 
                      d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N//2):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N//2):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    e1, e2, e3 = encoder_blocks
    d1, d2, d3 = decoder_blocks

    t_encoder_blocks = [e1, e2, e3, e3, e2, e1]
    t_decoder_blocks = [d1, d2, d3, d3, d2, d1]

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(t_encoder_blocks))
    decoder = Decoder(nn.ModuleList(t_decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters with xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # xaiver uniform initialization 

    return transformer