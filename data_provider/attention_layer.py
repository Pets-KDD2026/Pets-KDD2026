from math import sqrt

import torch
import torch.nn as nn

import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # print('############# FullAttention-1')
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # print(queries.shape)        # torch.Size([32, 7, 8, 128])
        # print(keys.shape)           # torch.Size([32, 7, 8, 128])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # print(scores.shape)         # torch.Size([32, 8, 7, 7])

        # print(self.mask_flag)       # True
        # print(attn_mask)            # None
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # print(type(attn_mask))  # <class 'utils.masking.TriangularCausalMask'>
            am = attn_mask.mask
            # print(am.shape)         # torch.Size([32, 1, 7, 7])
            scores.masked_fill_(am, -np.inf)

        # print(scale)                # 0.08838834764831843
        # print(scores.shape)         # torch.Size([32, 8, 7, 7])
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # print(A.shape)              # torch.Size([32, 8, 7, 7])
        # print(values.shape)         # torch.Size([32, 7, 8, 128])
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # print(V.shape)              # torch.Size([32, 7, 8, 128])

        # print('############# FullAttention-2')
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(mask_flag=False, scale=None, attention_dropout=0.1, output_attention=True)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # print('############# AttentionLayer-1')

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # print(type(self.query_projection))  # <class 'torch.nn.modules.linear.Linear'>
        # print(type(self.key_projection))    # <class 'torch.nn.modules.linear.Linear'>
        # print(type(self.value_projection))  # <class 'torch.nn.modules.linear.Linear'>
        # print(queries.shape)                # torch.Size([32, 7, 1024])
        # print(keys.shape)                   # torch.Size([32, 7, 1024])
        # print(values.shape)                 # torch.Size([32, 7, 1024])
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # print(queries.shape)                # torch.Size([32, 7, 8, 128])
        # print(keys.shape)                   # torch.Size([32, 7, 8, 128])
        # print(values.shape)                 # torch.Size([32, 7, 8, 128])

        # print(type(self.inner_attention))   # <class 'models.TimerBackbone.FullAttention'>
        # print(attn_mask)                    # None
        # print(tau)                          # None
        # print(delta)                        # None
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        # print(out.shape)                    # torch.Size([32, 7, 8, 128])
        # print(attn.shape)                   # torch.Size([32, 8, 7, 7])
        out = out.view(B, L, -1)
        # print(out.shape)                    # torch.Size([32, 7, 1024])

        # print(type(self.out_projection))    # <class 'torch.nn.modules.linear.Linear'>
        # print(out.shape)                    # torch.Size([32, 7, 1024])
        out = self.out_projection(out)
        # print(out.shape)                    # torch.Size([32, 7, 1024])

        # print('############# AttentionLayer-2')
        return out, attn