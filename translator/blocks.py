"""Blocks for the building the transformer"""

import numpy as np
import torch


class Padding:
    def __init__(self, seq_len, language):
        self.seq_len = seq_len
        assert language in ('english', 'german')
        self.language = language

    def __call__(self, x):
        #end of sentence (2) and ignore (-10) tokens are added
        if self.language == 'english':
            if len(x) < self.seq_len:
                padded_seq = x + [2 for i in range(len(x), self.seq_len)]
                return padded_seq #, len(x) + 1
            padded_seq = x[:self.seq_len - 1] + [2]
            return padded_seq #, self.seq_len
        if self.language == 'german':
            if len(x) < (self.seq_len - 1):
                padded_seq = [1] + x + [2] + [-10 for i in range(len(x) + 2, self.seq_len)]
                return padded_seq #, len(x) + 2
            padded_seq = [1] + x[:self.seq_len - 2] + [2]
            return padded_seq #, self.seq_len


class Embedding(torch.nn.Module):
    def __init__(self, emb_size, emb_dim):
        super().__init__()
        self.embedder = torch.nn.Embedding(emb_size, emb_dim)
        self.scale_factor = torch.nn.Parameter(torch.tensor(np.sqrt(emb_dim)), requires_grad=False)

    def forward(self, x):
        return self.scale_factor * self.embedder(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_dim, maxlen=1000):
        super().__init__()
        self.emb_dim = emb_dim
        self.maxlen = maxlen
        self.positional_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        signal = [[position / np.power(1e4, 2 * (i // 2) / self.emb_dim)
                   for i in range(self.emb_dim)] for position in range(self.maxlen)]
        signal = torch.Tensor(signal)
        signal[:, ::2] = torch.sin(signal[:, ::2])
        signal[:, 1::2] = torch.cos(signal[:, 1::2])
        signal = torch.nn.Parameter(signal.unsqueeze(0), requires_grad=False)
        return signal

    def forward(self, x):
        return x + self.positional_encoding[:, :x.shape[-2], :]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, masked=False):

        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.seq_len = seq_len
        self.masked = masked
        self.softmax = torch.nn.Softmax(dim=-1)

        if self.masked:
            self.mask = torch.nn.Parameter(torch.ones((1, self.seq_len, self.seq_len)) * -1e10,
                                           requires_grad=False)

        self.queries_projections = self._get_projections(self.d_k)
        self.keys_projections = self._get_projections(self.d_k)
        self.values_projections = self._get_projections(self.d_v)
        self.final_projection = torch.nn.Linear(self.h * self.d_v, self.d_model)

    def get_mask(self, n):
        return self.mask[:, :n, :n]

    def _get_projections(self, d_f):
        projections = []
        for _ in range(self.h):
            projections.append(torch.nn.Linear(self.d_model, d_f))
        return torch.nn.ModuleList(projections)

    def compute_simple_attention(self, q, k, v, l1=None, l2=None):
        a = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if l1 is not None:
            for i, ll1 in enumerate(l1):
                a[i, :, ll1:] = -1e10
        if l2 is not None:
            for i, ll2 in enumerate(l2):
                a[i, ll2:, :] = -1e10
        if self.masked:
            mask = self.get_mask(a.shape[1])
            a = a.tril() + mask.triu(1)
        a = self.softmax(a)
        return torch.matmul(a, v)

    def forward(self, q, k, v, l1=None, l2=None):
        heads = []
        for i in range(self.h):
            heads.append(self.compute_simple_attention(
                self.queries_projections[i](q),
                self.keys_projections[i](k),
                self.values_projections[i](v), l1, l2))
        return self.final_projection(torch.cat(heads, dim=-1))


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = torch.nn.Linear(self.d_model, self.d_ff)
        self.linear_2 = torch.nn.Linear(self.d_ff, self.d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.relu(self.linear_1(x)))
