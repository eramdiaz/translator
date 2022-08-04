"""Data utilities for the transformer"""

from random import random
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import Union, Iterable
from torch import LongTensor, ones, bool
from torch.utils.data import Dataset
from translator.tokenizer import load_tokenizer


class ItemGetter(Dataset):
    def __init__(self, data: Iterable, seq_len: int,
                 tokenizer: Union[str, Path, SentencePieceProcessor],
                 dataset_name: str = None,
                 prob_remove_punc: float = 0.8):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.tokenizer = self._get_tokenizer(tokenizer)
        self.dataset_name = dataset_name
        self.prob_remove_punct = prob_remove_punc

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
        sentence, translation = self.data[item]

        # Remove the point at the sentence with a probability of self.prob_rem_punc
        #for regularization.
        if sentence[-1] == '.' and translation[-1] == '.' and random() < self.prob_remove_punct:
            sentence = sentence[:-1]
            translation = translation[:-1]

        sentence = self.tokenizer.encode(sentence, out_type=int, add_bos=False, add_eos=True)
        translation = self.tokenizer.encode(translation, out_type=int, add_bos=True, add_eos=True)

        sen_mask = self.get_mask(len(sentence), len(sentence))
        tr_mask = self.get_mask(len(translation), len(translation))
        sentr_mask = self.get_mask(len(translation), len(sentence))

        sentence, translation = self.pad_sequence(sentence), self.pad_sequence(translation)
        sentence, translation = LongTensor(sentence), LongTensor(translation)

        return sentence, translation, sen_mask, sentr_mask, tr_mask

    def __len__(self):
        return len(self.data)
