"""Data utilities for the transformer"""

from torch.utils.data import Dataset
from torch import LongTensor
from translator.tokenizer import tokenizer


class ItemGetter(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.to_tensor = lambda x: LongTensor(x)

    def pad_sequence(self, seq):
        if len(seq) < self.seq_len:
            return seq + [self.tokenizer.pad_id() for _ in range(len(seq), self.seq_len)], [len(seq)]
        return seq[:self.seq_len - 1] + [self.tokenizer.eos_id()], [self.seq_len]

    def __getitem__(self, item):
        en_sentence, ger_sentence = self.data[item]

        en_sentence = self.tokenizer.encode(en_sentence, out_type=int, add_bos=False, add_eos=True)
        en_sentence, en_length = self.pad_sequence(en_sentence)
        en_sentence, en_length = self.to_tensor(en_sentence), self.to_tensor(en_length)

        ger_sentence = self.tokenizer.encode(ger_sentence, out_type=int, add_bos=True, add_eos=True)
        ger_sentence, ger_length = self.pad_sequence(ger_sentence)
        ger_sentence, ger_length = self.to_tensor(ger_sentence), self.to_tensor(ger_length)

        return (en_sentence, en_length),  (ger_sentence, ger_length)

    def __len__(self):
        return len(self.data)
