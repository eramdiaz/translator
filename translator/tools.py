"Tools for the transformer. "

import os
import pickle
from typing import Union, Tuple
from pathlib import Path
from torch import load
from torch.utils.data import Dataset
from translator.models import Transformer
from translator.train import Trainer
from translator.learning_rate import WarmUpLr


def load_model(path: Union[str, Path]):
    assert os.path.exists(path), f'The model {path} does not exist.'
    assert os.path.exists(f'{path}/tokenizer'), f'The model is missing the tokenizer associated.'
    assert os.path.exists(f'{path}/model.pt'), f'The model is missing the weights file'
    with open(f'{path}/tokenizer', 'r') as f:
        tokenizer_name = f.read()
    checkpoint = load(f'{path}/model.pt')
    params = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    model = Transformer(**params, tokenizer=tokenizer_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_standard_model():
    return Transformer(6, 'data/tokenizers/IWSLT2016-12000-bpe-0.model', 80, 512, 64, 64, 8, 2048)


def get_standard_trainer(
        model: Transformer = None,
        data: Tuple[Dataset, Dataset] = None,
        learning_rate: Union[float, WarmUpLr] = None,
        batch_size: int = 256,
        validation_freq: int = 1000,
        experiment: Union[str, Path] = None,
        predict_during_training: bool = True,
):
    if model is None:
        model = get_standard_model()
    if data is None:
        with open('data/IWSLT2016/train.pkl', 'rb') as f:
            train_samples = pickle.load(f)
        with open('data/IWSLT2016/valid.pkl', 'rb') as f:
            valid_samples = pickle.load(f)
    else:
        train_samples, valid_samples = data
    if learning_rate is None:
        learning_rate = 2e-4
        #learning_rate = WarmUpLr(4000, self.model.d_model)
    return Trainer(model, train_samples, valid_samples,
                   learning_rate, batch_size, experiment,
                   validation_freq, predict_during_training)
