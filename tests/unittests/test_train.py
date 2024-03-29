import pytest
import pickle
import torch
from datetime import datetime
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
BATCH_SIZE = 2
WARMUP_STEPS = 4000


@pytest.fixture(scope='module')
def mock_train():
    with open('tests/material/mock_train.pkl', 'rb') as f:
        mock_trainset = pickle.load(f)
    return mock_trainset


@pytest.fixture(scope='module')
def mock_valid():
    with open('tests/material/mock_valid.pkl', 'rb') as f:
        mock_validset = pickle.load(f)
    return mock_validset


class TestTrainer:
    @pytest.fixture(autouse=True)
    def init(self, request, mock_train, mock_valid, tmp_path):
        transformer = Transformer(N, 'tests/material/mock_tokenizer.model', SEQ_LEN, D_MODEL,
                                  D_K, D_V, H, D_FF)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transformer = transformer.to(device)
        lr_sch = WarmUpLr(WARMUP_STEPS, D_MODEL)
        exp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '/'
        request.cls.trainer = Trainer(transformer, mock_train, mock_valid,
                                      lr_sch, BATCH_SIZE, experiment=tmp_path / exp)

    def test_do_epoch(self):
        self.trainer.do_epoch()
