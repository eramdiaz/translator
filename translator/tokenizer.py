"""Tokenizer"""

import sentencepiece
from typing import Union
from os import remove
from pathlib import Path


def load_tokenizer(path: Union[str, Path]):
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.Load(str(path))
    tokenizer.name = str(path)
    return tokenizer


def train_tokenizer(corpus_file: str, model_prefix: str, vocab_size: int):
    sentencepiece.SentencePieceTrainer.Train(
        f'--input={corpus_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage={1.0} '
        '--control_symbols=<pad> '
        '--model_type=bpe '
    )
    remove(f'{model_prefix}.vocab')
