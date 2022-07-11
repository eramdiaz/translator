import pytest
import pickle
from torch.utils.data import Dataset
from translator.learning_rate import WarmUpLr
from translator.models import Transformer
from translator.train import Trainer

D_MODEL = 64
D_K = 8
D_V = 8
H = 8
SEQ_LEN = 32
D_FF = 128
N = 2
VOCAB_SIZE = 37000
BATCH_SIZE = 2
WARMUP_STEPS = 4000


class MyDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)


@pytest.fixture(scope='module')
def mock_train():
    with open('tests/material/mock_train.pkl', 'rb') as f:
        mock_trainset = pickle.load(f)
    return MyDataset(mock_trainset)


@pytest.fixture(scope='module')
def mock_valid():
    with open('tests/material/mock_valid.pkl', 'rb') as f:
        mock_validset = pickle.load(f)
    return MyDataset(mock_validset)


class TestTrainer:
    @pytest.fixture(autouse=True)
    def init(self, request, mock_train, mock_valid, tmp_path):
        transformer = Transformer(N, VOCAB_SIZE, SEQ_LEN, D_MODEL, D_K, D_V, H, D_FF)
        lr_sch = WarmUpLr(WARMUP_STEPS, D_MODEL)
        request.cls.trainer = Trainer(transformer, mock_train, mock_valid,
                                      lr_sch, BATCH_SIZE, experiment=tmp_path/'test.pt',
                                      predict_during_training=False)

    def test_do_epoch(self):
        self.trainer.do_epoch()
