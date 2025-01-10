import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class attentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # (sequence_length,batchsize,embed_dim)
        src = src.permute(2, 0, 1)  # 变为 [16, 1, 128]
        tar = tar.permute(2, 0, 1)  # 变为 [16, 1, 128]

        # Step 2: Multihead Attention
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]

        # Step 3: Add residual connection and apply layer normalization
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Step 4: Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
