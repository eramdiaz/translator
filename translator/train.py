"""Train the Transformer"""

import os
import torch
from time import time
from json import dump
from pathlib import Path
from datetime import datetime
from typing import Union, Iterable
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from translator.data import ItemGetter
from translator.learning_rate import wrap_lr


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_FOLDER = Path(__file__).resolve().parent.parent / 'checkpoints'
if not os.path.exists(CHECKPOINTS_FOLDER):
    os.makedirs(CHECKPOINTS_FOLDER)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_samples: Iterable,
            valid_samples: Iterable,
            learning_rate: Union[float, object],
            batch_size: int,
            experiment: Union[str, Path] = None,
            validation_freq: int = 500,
            dataset_name: str = None
    ):
        self.model = model
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.train_dataset = ItemGetter(self.train_samples, self.model.seq_len,
                                        self.model.tokenizer, dataset_name)
        self.valid_dataset = ItemGetter(self.valid_samples, self.model.seq_len,
                                        self.model.tokenizer, dataset_name)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size, True, num_workers=2)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size, True, num_workers=2)
        self.learning_rate = wrap_lr(learning_rate)
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.model.tokenizer.pad_id())
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-8, betas=(0.9, 0.98), eps=1e-9)
        self.max_bleu_score = 0.
        self.it = 0
        self.experiment = self._parse_experiment_path(experiment)
        self.validation_freq = validation_freq
        self.dataset_name = dataset_name

    def _parse_experiment_path(self, experiment):
        if experiment is None:
            experiment = CHECKPOINTS_FOLDER / datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        else:
            experiment = Path(experiment) if isinstance(experiment, str) else experiment
        if not os.path.isabs(experiment):
            experiment = CHECKPOINTS_FOLDER / experiment
        assert not os.path.exists(experiment), 'This experiment already exists. Please choose another name.'
        return experiment

    def train(self):
        print('Starting training\n')
        epochs = 0
        self.model.to(DEVICE)
        while True:
            print('Starting epoch %d\n' % epochs)
            self.do_epoch()
            epochs += 1

    def do_epoch(self):
        self.model.train()
        for enc_sentences, dec_sentences, enc_mask, decenc_mask, dec_mask in self.train_dataloader:
            start = time()
            enc_sentences, dec_sentences = enc_sentences.to(DEVICE), dec_sentences.to(DEVICE)
            enc_mask, decenc_mask, dec_mask = enc_mask.to(DEVICE), decenc_mask.to(DEVICE), dec_mask.to(DEVICE)
            self.optim.zero_grad()
            output = self.model(enc_sentences, dec_sentences, enc_mask, decenc_mask, dec_mask)
            loss = self.criterion(
                output[:, :-1, :].reshape(-1, output.shape[-1]), dec_sentences[:, 1:].reshape(-1)
            )
            loss.backward()
            self.optim.step()
            end = time()
            if self.it % 10 == 0:
                print(f'TRAIN iteration {self.it}; loss: {round(loss.item(), 4)}; '
                      f'lr: {self.optim.param_groups[0]["lr"]}; '
                      f'iteration time: {round(end - start, 4)}\n')

            if self.it % self.validation_freq == 0:
                self.valid_epoch()
                self.model.train()
            self.it += 1
            self.learning_rate(self.it, self.optim.param_groups)

    def valid_epoch(self):
        print('Validating...\n')
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.
            eval_it = 0
            total_bleu = 0.
            for enc_sentences, dec_sentences, enc_mask, decenc_mask, dec_mask in self.valid_dataloader:
                start = time()
                enc_sentences, dec_sentences = enc_sentences.to(DEVICE), dec_sentences.to(DEVICE)
                enc_mask, decenc_mask, dec_mask = enc_mask.to(DEVICE), decenc_mask.to(DEVICE), dec_mask.to(DEVICE)
                encoded = self.model.encode(enc_sentences, enc_mask)
                output = self.model.decode(encoded, dec_sentences, decenc_mask, dec_mask)
                loss = self.criterion(
                    output[:, :-1, :].reshape(-1, output.shape[-1]), dec_sentences[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
                bleu = self.compute_bleu_score(encoded, dec_sentences, decenc_mask[:, :1,:])
                end = time()
                print(f'VALID iteration {eval_it}; loss: {round(loss.item(), 4)}; '
                      f'iteration time: {round(end - start, 4)}\n')
                total_bleu += bleu
                eval_it += 1
            total_loss = total_loss / eval_it
            total_bleu = total_bleu / eval_it
            print('=' * 140 + '\n' +
                  f'VALID epoch; valid loss: {round(total_loss, 4)}; bleu score: '
                  f'{round(total_bleu, 4)}\n' +
                  '=' * 140 + '\n')
            if total_bleu > self.max_bleu_score:
                self.max_bleu_score = total_bleu
                self.save_model()

    def compute_bleu_score(self, enc_output, targets, mask):
        candidates = self.model.predict(enc_output, mask=mask, already_encoded=True,
                                        max_len=self.model.seq_len)
        candidates = [cand.split(' ') for cand in candidates]
        candidates = [[split for split in cand if split] for cand in candidates]
        references = self.model.tokenizer.decode(targets.to('cpu').tolist())
        references = [[ref.split(' ')] for ref in references]
        return bleu_score(candidates, references, max_n=2, weights=[0.5, 0.5])

    def save_model(self):
        print('Saving...\n')
        if not os.path.exists(self.experiment):
            os.makedirs(self.experiment)
        info = {'tokenizer': self.model.tokenizer.name, 'bleu_score': round(self.max_bleu_score, 4),
                'dataset': self.train_dataset.dataset_name}
        with open(self.experiment / 'info.json', 'w') as f:
            dump(info, f)
        checkpoint = {
            'n': self.model.n, 'seq_len': self.model.seq_len, 'd_model': self.model.d_model,
            'd_k': self.model.d_k, 'd_v': self.model.d_v, 'h': self.model.h,
            'd_ff': self.model.d_ff, 'state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, self.experiment / 'model.pt')
