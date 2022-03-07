"""Train our tokenizer"""

import sentencepiece
from pathlib import Path

CORPUS_FILE = Path(__file__).resolve().parent.parent / 'data' / 'train_subset.txt'
MODEL_PREFIX = 'tokenizer'
VOCAB_SIZE = 37000

if __name__ == '__main__':
    sentencepiece.SentencePieceTrainer.Train(
       f'--input={str(CORPUS_FILE)} '
       f'--model_prefix={MODEL_PREFIX} '
       f'--vocab_size={VOCAB_SIZE} '
       f'--character_coverage={1.0} '
       '--control_symbols=<pad> '
       '--model_type=bpe '
    )
