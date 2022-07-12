"""Build of the transformer architecture"""

from translator.blocks import *
from translator.tokenizer import tokenizer
from torch import nn


class EncoderCell(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, d_ff):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, False)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask=None):
        x = self.layernorm1(self.dropout1(self.attention_layer(q, k, v, mask, mask)) + q)
        return self.layernorm2(self.dropout2(self.feedforward(x)) + x)


class DecoderCell(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, True)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, False)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, d_ff)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, q, dec_k, dec_v, enc_k, enc_v, i_mask=None, o_mask=None):
        x = self.layernorm1(self.dropout1(
            self.self_attention(q, dec_k, dec_v, o_mask, o_mask)) + q)
        x = self.layernorm2(self.dropout2(
            self.enc_dec_attention(x, enc_k, enc_v, i_mask, o_mask)) + x)
        return self.layernorm3(self.dropout3(self.feedforward(x)) + x)


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
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.final_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

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

    def encode(self, inputs, mask=None):
        inputs = self.dropout1(self.positional_encoding(self.embedding(inputs)))
        for module in self.encoder:
            inputs = module(inputs, inputs, inputs, mask)
        return inputs

    def decode(self, inputs, outputs, i_mask=None, o_mask=None):
        outputs = self.dropout2(self.positional_encoding(self.embedding(outputs)))
        for module in self.decoder:
            outputs = module(outputs, outputs, outputs, inputs, inputs,
                             i_mask, o_mask)
        return self.final_projection(outputs)

    def forward(self, inputs, outputs, i_mask=None, o_mask=None):
        inputs = self.encode(inputs, i_mask)
        return self.decode(inputs, outputs, i_mask, o_mask)

    def predict(self, sentence, tokenizer=tokenizer, max_len=None):
        start_token, end_token = tokenizer.bos_id(), tokenizer.eos_id()
        max_len = self.positional_encoding.max_len if max_len is None else max_len
        device = next(self.parameters()).device.type
        sequence = torch.LongTensor([start_token]).to(device)
        input_ = tokenizer.encode(sentence, out_type=int, add_bos=False, add_eos=True)
        self.eval()
        input_ = self.encode(torch.LongTensor(input_).to(device))
        for _ in range(max_len):
            output = self.softmax(self.decode(input_, sequence))
            #prediction = torch.multinomial(output[0, -1, :], 1)
            prediction = output[0, -1, :].topk(1)[1]
            if prediction.item() == end_token:
                break
            sequence = torch.cat((sequence, prediction), -1)
        sequence = sequence.to('cpu')
        return tokenizer.decode(sequence[1:].tolist())
