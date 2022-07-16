"""Blocks for the building the transformer"""

import numpy as np
import torch
from torch import nn

MAX_LEN = 1000


def initialize_weight(x):
    if isinstance(x, nn.ModuleList):
        for y in x:
            initialize_weight(y)
    else:
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Embedding(nn.Module):
    def __init__(self, emb_size, emb_dim, do_weight_init=False):
        super().__init__()
        self.embedder = nn.Embedding(emb_size, emb_dim)
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.do_weight_init = do_weight_init
        if self.do_weight_init:
            nn.init.normal_(self.embedder.weight, mean=0, std=emb_dim**-0.5)
        self.scale_factor = emb_dim ** 0.5

    def forward(self, x):
        return self.scale_factor * self.embedder(x)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=MAX_LEN):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.positional_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        signal = [[position / np.power(1e4, 2 * (i // 2) / self.emb_dim)
                   for i in range(self.emb_dim)] for position in range(self.max_len)]
        signal = torch.Tensor(signal)
        signal[:, ::2] = torch.sin(signal[:, ::2])
        signal[:, 1::2] = torch.cos(signal[:, 1::2])
        signal = nn.Parameter(signal.unsqueeze(0), requires_grad=False)
        return signal

    def forward(self, x):
        return x + self.positional_encoding[:, :x.shape[-2], :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, masked=False,
                 do_weight_init=False, max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.scaling_factor = np.sqrt(self.d_k)
        self.seq_len = seq_len
        self.masked = masked
        self.do_weight_init = do_weight_init
        self.softmax = nn.Softmax(dim=-1)

        if self.masked:
            self.mask = nn.Parameter(torch.ones((1, max_len, max_len)) * -1e10,
                                           requires_grad=False)

        self.queries_projections = self._get_projections(self.d_k)
        self.keys_projections = self._get_projections(self.d_k)
        self.values_projections = self._get_projections(self.d_v)
        self.final_projection = nn.Linear(self.h * self.d_v, self.d_model, bias=False)

        if self.do_weight_init:
            initialize_weight(self.queries_projections)
            initialize_weight(self.keys_projections)
            initialize_weight(self.values_projections)
            initialize_weight(self.final_projection)

    def get_mask(self, n):
        return self.mask[:, :n, :n]

    def _get_projections(self, d_f):
        projections = []
        for _ in range(self.h):
            projections.append(nn.Linear(self.d_model, d_f, bias=False))
        return nn.ModuleList(projections)

    def compute_simple_attention(self, q, k, v, pad_mask_1=None, pad_mask_2=None):
        w = torch.matmul(q, k.transpose(-2, -1)) / self.scaling_factor
        if pad_mask_1 is not None:
            for i, pm1 in enumerate(pad_mask_1):
                w[i, :, pm1:] = -1e10
        if pad_mask_2 is not None:
            for i, pm2 in enumerate(pad_mask_2):
                w[i, pm2:, :] = -1e10
        if self.masked:
            mask = self.get_mask(w.shape[-2])
            w = w.tril() + mask.triu(1)
        w = self.softmax(w)
        return torch.matmul(w, v)

    def forward(self, q, k, v, pad_mask_1=None, pad_mask_2=None):
        heads = []
        for i in range(self.h):
            heads.append(self.compute_simple_attention(
                self.queries_projections[i](q),
                self.keys_projections[i](k),
                self.values_projections[i](v),
                pad_mask_1, pad_mask_2))
        return self.final_projection(torch.cat(heads, dim=-1))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, do_weight_init=False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.do_weight_init = do_weight_init
        self.linear_1 = nn.Linear(self.d_model, self.d_ff)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model)
        if self.do_weight_init:
            initialize_weight(self.linear_1)
            initialize_weight(self.linear_2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.relu(self.linear_1(x)))
