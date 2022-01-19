"""Data utilities for the transformer"""

import sentencepiece
from translator.blocks import Padding
from torch.utils.data import Dataset
from torch import LongTensor


class ItemGetter(Dataset):
    def __init__(self, eng_sentences, ger_sentences, seq_len):
        assert len(eng_sentences) == len(ger_sentences)
        self.eng_sentences = eng_sentences
        self.ger_sentences = ger_sentences
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load('tokenizer.model')
        self.tokenizer.dic = {i: self.tokenizer.decode([i]) for i in
                              range(self.tokenizer.vocab_size())}
        self.eng_padding = Padding(seq_len, 'english')
        self.ger_padding = Padding(seq_len, 'german')
        self.to_tensor = lambda x: LongTensor(x)

    def __getitem__(self, item):
        x = self.eng_sentences[item]
        x = self.tokenizer.encode_as_ids(x)
        x = self.english_padding(x)
        x = self.to_tensor(x)

        y = self.ger_sentences[item]
        y = self.encode_as_ids(y)
        y = self.ger_padding(y)
        y = self.to_tensor(y)

        return x, y

    def __len__(self):
        return len(self.eng_sentences)
