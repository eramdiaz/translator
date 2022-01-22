"""Tokenizer"""

import sentencepiece

tokenizer = sentencepiece.SentencePieceProcessor()
tokenizer_path = '/'.join(__file__.split('/', -1)[:-1]) + '/tokenizer.model'
tokenizer.Load(tokenizer_path)
tokenizer.dic = {i: tokenizer.decode([i]) for i in range(tokenizer.vocab_size())}
