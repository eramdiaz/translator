from torchtext.datasets import IWSLT2016
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import ConcatDataset
from translator.models import Transformer
from translator.learning_rate import WarmUpLr
from translator.train import Trainer


N = 6
D_MODEL = 512
SEQ_LEN = 80
H = 8
D_K = D_V = 64
D_FF = 2048
tokenizer_path = 'data/tokenizers/IWSLT2016-12000-bpe-0.model'
WARMUP_STEPS = 4000
BATCH_SIZE = 256


def main():
    train_iter, valid_iter, test_iter = IWSLT2016(language_pair=('en', 'de'))
    train_samples = to_map_style_dataset(train_iter)
    valid_samples = to_map_style_dataset(valid_iter)
    test_samples = to_map_style_dataset(test_iter)
    translator = Transformer(N, tokenizer_path, SEQ_LEN, D_MODEL, D_K, D_V, H, D_FF)
    #lr_sch = WarmUpLr(WARMUP_STEPS, D_MODEL)
    lr_sch = 2e-4
    trainer = Trainer(translator, train_samples, ConcatDataset([valid_samples, test_samples]),
                      lr_sch, BATCH_SIZE, validation_freq=500)
    trainer.train()


if __name__ == '__main__':
    main()
