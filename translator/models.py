"""Build of the transformer architecture"""

from translator.blocks import *
from translator.tokenizer import tokenizer
from torch import nn


class EncoderCell(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, d_ff, do_weight_init=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.do_weight_init = do_weight_init
        self.attention_layer = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, False, do_weight_init)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, d_ff, do_weight_init)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask=None):
        x = self.layernorm1(self.dropout1(self.attention_layer(q, k, v, mask, mask)) + q)
        return self.layernorm2(self.dropout2(self.feedforward(x)) + x)


class DecoderCell(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, seq_len, d_ff, do_weight_init=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.do_weight_init = do_weight_init
        self.self_attention = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, True, do_weight_init)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(d_model, d_k, d_v, h, seq_len, False, do_weight_init)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.feedforward = FeedForward(d_model, d_ff, do_weight_init)
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
    def __init__(self, n, vocab_size, seq_len, d_model, d_k, d_v, h, d_ff, do_weight_init=False):
        super().__init__()
        self.n = n
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.do_weight_init = do_weight_init
        self.embedding = Embedding(self.vocab_size, self.d_model, self.do_weight_init)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.final_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if self.do_weight_init:
            initialize_weight(self.final_projection)
        self.softmax = nn.Softmax(dim=-1)

    def _get_encoder(self):
        encoder = []
        for _ in range(self.n):
            encoder.append(EncoderCell(self.d_model, self.d_k, self.d_v, self.h,
                                       self.seq_len, self.d_ff, self.do_weight_init))
        return nn.ModuleList(encoder)

    def _get_decoder(self):
        decoder = []
        for _ in range(self.n):
            decoder.append(DecoderCell(self.d_model, self.d_k, self.d_v, self.h,
                                       self.seq_len, self.d_ff, self.do_weight_init))
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

    def predict(self, input_, already_encoded=False, tokenizer=tokenizer, max_len=None):
        start_token, end_token = tokenizer.bos_id(), tokenizer.eos_id()
        max_len = self.positional_encoding.max_len if max_len is None else max_len
        device = next(self.parameters()).device.type
        self.eval()
        if not already_encoded:
            input_ = tokenizer.encode(input_, out_type=int, add_bos=False, add_eos=True)
            input_ = self.encode(torch.LongTensor(input_).to(device))
        prediction = torch.ones((len(input_), 1), dtype=torch.int64) * start_token
        prediction = prediction.to(device)
        for _ in range(max_len):
            output = self.softmax(self.decode(input_, prediction))
            #pred = torch.multinomial(output[:, -1, :], 1)
            pred = output[:, -1, :].topk(1)[1]
            if len(pred) == 1:
                if pred.item() == end_token:
                    break
            prediction = torch.cat((prediction, pred), -1)
        prediction = prediction.to('cpu').tolist()
        if len(pred) == 1:
            return tokenizer.decode(prediction)[0]
        for i, sent in enumerate(prediction):
            try:
                stop_index = sent.index(end_token)
            except ValueError:
                pass
            else:
                prediction[i] = sent[:stop_index]
        return tokenizer.decode(prediction)
