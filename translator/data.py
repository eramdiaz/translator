"""Data utilities for the transformer"""

from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import Union, Iterable
from torch import LongTensor, ones, bool
from torch.utils.data import Dataset
from translator.tokenizer import load_tokenizer


class ItemGetter(Dataset):
    def __init__(self, data: Iterable, seq_len: int,
                 tokenizer: Union[str, Path, SentencePieceProcessor]):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.tokenizer = self._get_tokenizer(tokenizer)

    @staticmethod
    def _get_tokenizer(tokenizer):
        if isinstance(tokenizer, SentencePieceProcessor):
            return tokenizer
        elif isinstance(tokenizer, str) or isinstance(tokenizer, Path):
            return load_tokenizer(tokenizer)
        else:
            raise ValueError('tokenizer argument must be a pathlib.Path or a string object '
                             'representing the path of the tokenizer to load or a '
                             'sentencepiece.SentencePieceProcessor object.')

    def pad_sequence(self, seq):
        if len(seq) < self.seq_len:
            return seq + [self.tokenizer.pad_id() for _ in range(len(seq), self.seq_len)]
        return seq[:self.seq_len - 1] + [self.tokenizer.eos_id()]

    def get_mask(self, hor, ver):
        mask = ones(self.seq_len, self.seq_len, dtype=bool)
        mask[:hor, :ver] = 0
        return mask

    def __getitem__(self, item):
        en_sentence, de_sentence = self.data[item]
        en_sentence = self.tokenizer.encode(en_sentence, out_type=int, add_bos=False, add_eos=True)
        de_sentence = self.tokenizer.encode(de_sentence, out_type=int, add_bos=True, add_eos=True)

        en_mask = self.get_mask(len(en_sentence), len(en_sentence))
        de_mask = self.get_mask(len(de_sentence), len(de_sentence))
        deen_mask = self.get_mask(len(de_sentence), len(en_sentence))

        en_sentence, de_sentence = self.pad_sequence(en_sentence), self.pad_sequence(de_sentence)
        en_sentence, de_sentence = LongTensor(en_sentence), LongTensor(de_sentence)

        return en_sentence, de_sentence, en_mask, deen_mask, de_mask

    def __len__(self):
        return len(self.data)
