"""Tokenizer"""

import sentencepiece
from pathlib import Path


tokenizer = sentencepiece.SentencePieceProcessor()
tokenizer_path = Path(__file__).resolve().parent / 'tokenizer.model'
tokenizer.Load(str(tokenizer_path))
tokenizer.dic = {i: tokenizer.decode([i]) for i in range(tokenizer.vocab_size())}
