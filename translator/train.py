"""Train the Transformer"""

import os
import torch
from time import time
from uuid import uuid4
from typing import Union, Tuple, Iterable, Any
from torch.utils.data import DataLoader
from translator.data import ItemGetter
from translator.learning_rate import wrap_lr


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_FOLDER = '/'.join(__file__.split('/')[:-2]) + '/checkpoints'
assert os.path.exists(CHECKPOINTS_FOLDER), \
    'Create a checkpoints folder in translator for saving the model.'
CHECKPOINT_PATH = CHECKPOINTS_FOLDER + '/' + str(uuid4()) + '.pth'


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_samples: Tuple[Iterable[str], Iterable[str]],
            valid_samples: Tuple[Iterable[str], Iterable[str]],
            learning_rate: Union[float, object],
            batch_size: int,
            criterion: torch.nn.Module = None,
            optim: torch.nn.Module = None,
    ):
        self.model = model
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.train_set = ItemGetter(*self.train_samples, self.model.seq_len)
        self.valid_set = ItemGetter(*self.valid_samples, self.model.seq_len)
        self.train_loader = DataLoader(self.train_set, batch_size, True, num_workers=2)
        self.valid_loader = DataLoader(self.valid_set, batch_size, True, num_workers=2)
        import pdb; pdb.set_trace()
        self.learning_rate = wrap_lr(learning_rate)
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=3) if criterion is None else criterion
        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=1e-7, betas=(0.9, 0.98), eps=1e-09) if optim is None else optim
        self.min_valid_loss = 1e10

    def loss_function(self, predictions, targets):
        total_loss = 0.
        for pred, target in zip(predictions, targets):
            loss_item = self.criterion(pred[:-1, :], target[1:])
            total_loss += loss_item
        return total_loss / predictions.shape[0]

    def train(self):
        epochs = 0
        it = 0.
        self.model.eval()
        while True:
            print('Start epoch %d' % epochs)
            self.model.train()
            for (en_sentences, en_mask), (ger_sentences, ger_mask) in self.train_loader:
                start = time()
                en_sentences, ger_sentences = en_sentences.to(DEVICE), ger_sentences.to(DEVICE)
                en_mask, ger_mask = en_mask.to(DEVICE), ger_mask.to(DEVICE)
                self.optim.zero_grad()
                output = self.model.forward(en_sentences, ger_sentences, en_mask, ger_mask)
                loss = self.loss_function(output, ger_sentences)
                loss.backward()
                self.optim.step()
                end = time()
                print(f'TRAIN iteration {it}; loss: {round(loss.item(), 4)}; '
                      f'lr: {self.optim.param_groups[0]["lr"]}; '
                      f'iteration time: {round(end - start, 4)}')

                if it % 50 == 0 and it > 0:
                    self.valid_epoch()
                    self.model.train()
                it += 1
                self.learning_rate(it, self.optim.param_groups)
            epochs += 1

    def valid_epoch(self):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.
            eval_it = 0.
            for (en_sentences, en_mask), (ger_sentences, ger_mask) in self.valid_loader:

                if eval_it == 0:
                    import pdb; pdb.set_trace()
                    for el in self.train_samples[0]:
                        print(el)
                        print(self.model.predict(el), '\n')

                    train_sentence = self.train_samples[0][3]
                    valid_sentence = self.valid_samples[0][4]
                    print('\n' + train_sentence + '\n')
                    print(self.model.predict(train_sentence), '\n')

                    print(valid_sentence + '\n')
                    print(self.model.predict(valid_sentence), '\n')

                en_sentences, ger_sentences = en_sentences.to(DEVICE), ger_sentences.to(DEVICE)
                en_mask, ger_mask = en_mask.to(DEVICE), ger_mask.to(DEVICE)
                output = self.model.forward(en_sentences, ger_sentences, en_mask, ger_mask)
                loss = self.loss_function(output, ger_sentences)
                total_loss += loss.item()
                eval_it += 1
            total_loss = total_loss / eval_it
            print(f'VALID iteration; valid_loss: {round(total_loss, 4)};')
            if total_loss < self.min_valid_loss:
                self.min_valid_loss = total_loss
                self.save_model()

    def save_model(self):
        print('Saving...')
        checkpoint = {
            'n': self.model.n, 'vocab_size': self.model.vocab_size, 'seq_len': self.model.seq_len,
            'd_model': self.model.d_model, 'd_k': self.model.d_k, 'd_v': self.model.d_v,
            'h': self.model.h, 'd_ff': self.model.d_ff, 'state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
