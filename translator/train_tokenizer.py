"""Train our tokenizer."""

import sentencepiece

CORPUS_FILE = 'data/train_subset.txt'
MODEL_PREFIX = 'tokenizer'
VOCAB_SIZE = 37000

if __name__ == '__main__':
   sentencepiece.SentencePieceTrainer.Train(
      f'--input={CORPUS_FILE} '
      f'--model_prefix={MODEL_PREFIX} '
      f'--vocab_size={VOCAB_SIZE} '
      f'--character_coverage={1.0} '
      '--model_type=bpe '
   )
