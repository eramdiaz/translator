"""Train our tokenizer"""

import sentencepiece

CORPUS_FILE = '/'.join(__file__.split('/', -2)[:-2]) + '/data/train_subset.txt'
MODEL_PREFIX = 'tokenizer'
VOCAB_SIZE = 37000

if __name__ == '__main__':
    sentencepiece.SentencePieceTrainer.Train(
       f'--input={CORPUS_FILE} '
       f'--model_prefix={MODEL_PREFIX} '
       f'--vocab_size={VOCAB_SIZE} '
       f'--character_coverage={1.0} '
       '--control_symbols=<pad> '
       '--model_type=bpe '
    )
