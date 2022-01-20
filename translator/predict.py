"""Predict with out model"""

import torch
from translator.tokenizer import tokenizer


def predict(sentence, checkpoint, maxlen=5000):
    start_token = 1
    end_token = 2
    sequence = torch.LongTensor([start_token])
    it = 0
    model = infer_translator
    tokenized_input = tokenizer.encode_as_ids(sentence)
    while True:
        output = model(tokenized_input, sequence)
        prediction = output[:, -1, :].topk(1)[1]
        if prediction.item() == end_token:
            break
        torch.cat((sequence, prediction), -1)
        sequence.append(prediction)
        it += 1
        if it == maxlen:
            break
    return tokenizer.decode(sequence[1:].tolist())
