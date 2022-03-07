"""Train our tokenizer"""

import sentencepiece


def train_tokenizer(corpus_file, model_prefix, vocab_size):
    sentencepiece.SentencePieceTrainer.Train(
       f'--input={corpus_file} '
       f'--model_prefix={model_prefix} '
       f'--vocab_size={vocab_size} '
       f'--character_coverage={1.0} '
       '--control_symbols=<pad> '
       '--model_type=bpe '
    )
