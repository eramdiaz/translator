import pandas as pd
from translator.models import Transformer
from translator.learning_rate import WarmUpLr
from translator.train import Trainer


N = 6
D_MODEL = 512
SEQ_LEN = 96
H = 8
D_K = D_V = 64
D_FF = 2048
VOCAB_SIZE = 37000
WARMUP_STEPS = 4000
BATCH_SIZE = 256
PROJECT_FOLDER = '/'.join(__file__.split('/')[:-1])


def main():
    en_train = pd.read_csv(f'{PROJECT_FOLDER}/data/en_train.txt', header=None)[0].tolist() #nrows
    ger_train = pd.read_csv(f'{PROJECT_FOLDER}/data/ger_train.txt', header=None)[0].tolist()
    en_valid = pd.read_csv(f'{PROJECT_FOLDER}/data/en_valid.txt', header=None)[0].tolist()
    ger_valid = pd.read_csv(f'{PROJECT_FOLDER}/data/ger_valid.txt', header=None)[0].tolist()
    translator = Transformer(N, VOCAB_SIZE, SEQ_LEN, D_MODEL, D_K, D_V, H, D_FF)
    lr_sch = WarmUpLr(WARMUP_STEPS, D_MODEL)
    #lr_sch = 9e-5
    trainer = Trainer(translator, (en_train, ger_train), (en_valid, ger_valid), lr_sch, BATCH_SIZE)
    trainer.train()


if __name__ == '__main__':
    main()
