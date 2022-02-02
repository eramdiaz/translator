"""Build of the transformer architecture"""

from translator.blocks import *
from translator.tokenizer import tokenizer
from torch import nn


class EncoderCell(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, d_ff):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, False)
        self.layernorm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layernorm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, lengths=None):
        x = self.layernorm_1(self.dropout(self.attention_layer(q, k, v, lengths, lengths)) + q)
        return self.layernorm_2(self.dropout(self.feedforward(x)) + x)


class DecoderCell(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, True)
        self.layernorm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, False)
        self.layernorm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layernorm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, dec_k, dec_v, enc_k, enc_v, e_lengths=None, d_lengths=None):
        x = self.layernorm_1(self.dropout(
            self.self_attention(q, dec_k, dec_v, d_lengths, d_lengths)) + q)
        x = self.layernorm_2(self.dropout(
            self.enc_dec_attention(x, enc_k, enc_v, e_lengths, d_lengths)) + x)
        return self.layernorm_3(self.dropout(self.feedforward(x)) + x)


class Transformer(nn.Module):
    def __init__(self, n, vocab_size, seq_len, d_model, d_k, d_v, h, d_ff):
        super().__init__()
        self.n = n
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.final_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)

        torch.nn.init.xavier_uniform_(self.final_projection.weight, LINEAR_GAIN)

    def _get_encoder(self):
        encoder = []
        for _ in range(self.n):
            encoder.append(EncoderCell(self.d_model, self.d_k, self.d_v, self.h,
                                       self.seq_len, self.d_ff))
        return nn.ModuleList(encoder)

    def _get_decoder(self):
        decoder = []
        for _ in range(self.n):
            decoder.append(DecoderCell(self.d_model, self.d_k, self.d_v, self.h,
                                       self.seq_len, self.d_ff))
        return nn.ModuleList(decoder)

    def forward(self, inputs, outputs, inputs_lengths=None, outputs_lengths=None):
        inputs = self.dropout(self.positional_encoding(self.embedding(inputs)))
        for module in self.encoder:
            inputs = module(inputs, inputs, inputs, inputs_lengths)
        outputs = self.dropout(self.positional_encoding(self.embedding(outputs)))
        for module in self.decoder:
            outputs = module(outputs, outputs, outputs, inputs, inputs,
                             inputs_lengths, outputs_lengths)
        return self.final_projection(outputs)

    def predict(self, sentence, tokenizer=tokenizer, start_token=1, end_token=2):
        # TODO: add ' . ', ' ? ', ' ! ' like in train data if the sentence doesn't finish with them?
        sequence = torch.LongTensor([start_token])
        it = 0
        tokenized_input = torch.LongTensor(tokenizer.encode_as_ids(sentence) + [end_token])
        self.eval()
        while True:
            output = self.forward(tokenized_input, sequence)
            prediction = output[:, -1, :].topk(1)[1].squeeze(0)
            if prediction.item() == end_token:
                break
            sequence = torch.cat((sequence, prediction), -1)
            it += 1
            if it == self.positional_encoding.max_len:
                break
        return tokenizer.decode(sequence[1:].tolist())
