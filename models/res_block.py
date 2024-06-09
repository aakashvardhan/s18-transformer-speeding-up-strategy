import math
import torch
import torch.nn as nn
from models.layer_norm import LayerNormalization


class ResidualConnection(nn.Module):
    
        def __init__(self, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization()
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))