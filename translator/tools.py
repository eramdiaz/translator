"Tools for the transformer. "

import os
import pickle
import torch
import gdown
from typing import Union, Tuple
from json import load
from pathlib import Path
from torch.utils.data import Dataset
from translator.models import Transformer
from translator.train import Trainer
from translator.learning_rate import WarmUpLr


MODEL_ID = {
    'en-de-base-2807': '1J5LwGA1zHuUh8HOUgNY2E5-WzZlbIGD9'
}


def load_model(path: Union[str, Path]):
    assert os.path.exists(path), f'The model {path} does not exist.'
    assert os.path.exists(f'{path}/info.json'), f'The model is missing the info file associated.'
    with open(f'{path}/info.json', 'r') as f:
        info = load(f)
    if not os.path.exists(f'{path}/model.pt'):
        url = f"https://drive.google.com/uc?id={MODEL_ID[info['model']]}"
        output = f'{path}/model.pt'
        gdown.download(url, output, quiet=False)
    dataset_name = info['dataset'] if info['dataset'] is not None else 'an unknown dataset'
    print(f'Loading model {path}, which achieves a bleu score of {info["bleu_score"]} on'
          f'{dataset_name}')
    tokenizer_path = Path(__file__).resolve().parent.parent / info['tokenizer']
    checkpoint = torch.load(f'{path}/model.pt', map_location='cpu')
    params = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    model = Transformer(**params, tokenizer=tokenizer_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_standard_model():
    tokenizer_path = Path(__file__).resolve().parent.parent / \
                     'data/tokenizers/IWSLT2016-12000-bpe-0.model'
    return Transformer(6, tokenizer_path, 80, 512, 64, 64, 8, 2048)


def get_standard_trainer(
        model: Transformer = None,
        data: Tuple[Dataset, Dataset] = None,
        learning_rate: Union[float, WarmUpLr] = None,
        batch_size: int = 256,
        validation_freq: int = 500,
        experiment: Union[str, Path] = None,
        dataset_name: str = None
):
    if model is None:
        model = get_standard_model()
    if data is None:
        dataset_name = 'IWSLT2016' if dataset_name is None else dataset_name
        data_folder = Path(__file__).resolve().parent.parent / 'data/IWSLT2016'
        with open(data_folder / 'train.pkl', 'rb') as f:
            train_samples = pickle.load(f)
        with open(data_folder / 'valid.pkl', 'rb') as f:
            valid_samples = pickle.load(f)
    else:
        train_samples, valid_samples = data
    if learning_rate is None:
        learning_rate = WarmUpLr(4000, model.d_model)
    return Trainer(model, train_samples, valid_samples,
                   learning_rate, batch_size, experiment,
                   validation_freq, dataset_name)
