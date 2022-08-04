import pytest
import pickle
from translator.models import Transformer
from translator.tools import *


@pytest.fixture()
def mock_folder(tmp_path):
    return tmp_path / 'mock_model/'


def test_get_standard_trainer(mock_folder):
    with open("tests/material/mock_train.pkl", "rb") as f:
        mock_trainset = pickle.load(f)
    with open("tests/material/mock_valid.pkl", "rb") as f:
        mock_validset = pickle.load(f)
    model = Transformer(2, "tests/material/mock_tokenizer.model", 32, 64, 8, 8, 4, 128)
    trainer = get_standard_trainer(model, (mock_trainset, mock_validset), 0.02, 2,
                                   mock_folder, dataset_name = "mock_data")
    trainer.max_bleu_score = 0.0024
    trainer.save_model()


def test_load_model(mock_folder):
    model_folder = mock_folder.parent.parent / ('test_get_standard_trainer0/' + 'mock_model/')
    load_model(model_folder)
