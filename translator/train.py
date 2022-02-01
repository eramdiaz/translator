"""Train the Transformer"""

import os
import pandas as pd
import torch
from time import time
from uuid import uuid4
from torch.utils.data import DataLoader
from translator.data import ItemGetter
from translator.models import Transformer
from translator.learning_rate import WarmUpLr


N = 2
D_MODEL = 512
SEQ_LEN = 96
H = 8
D_K = D_V = 64
D_FF = 2048
VOCAB_SIZE = 37000
WARMUP_STEPS = 4000
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_FOLDER = '/'.join(__file__.split('/', -2)[:-2]) + '/checkpoints'
assert os.path.exists(CHECKPOINTS_FOLDER), \
    'Create a checkpoints folder in translator for saving the model.'
CHECKPOINT_PATH = CHECKPOINTS_FOLDER + '/' + str(uuid4()) + '.pth'

translator = Transformer(N, VOCAB_SIZE, SEQ_LEN, D_MODEL, D_K, D_V, H, D_FF)

PROJECT_FOLDER = '/'.join(__file__.split('/')[:-2])
EN_TRAIN = pd.read_csv(f'{PROJECT_FOLDER}/data/en_train.txt', delimiter='\n', header=None)[0].tolist() #nrows
GER_TRAIN = pd.read_csv(f'{PROJECT_FOLDER}/data/ger_train.txt', delimiter='\n', header=None)[0].tolist()
EN_VALID = pd.read_csv(f'{PROJECT_FOLDER}/data/en_valid.txt', delimiter='\n', header=None)[0].tolist()
GER_VALID = pd.read_csv(f'{PROJECT_FOLDER}/data/ger_valid.txt', delimiter='\n', header=None)[0].tolist()

train_dataset = ItemGetter(EN_TRAIN, GER_TRAIN, SEQ_LEN)
valid_dataset = ItemGetter(EN_VALID, GER_VALID, SEQ_LEN)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

loss = torch.nn.CrossEntropyLoss(ignore_index=3)


def loss_function(predictions, targets):
    total_loss = 0.
    for pred, target in zip(predictions, targets):
        loss_item = loss(pred[:-1, :], target[1:])
        total_loss += loss_item
    return total_loss / predictions.shape[0]


def train():
    optim = torch.optim.Adam(translator.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-09)
    lr_schedule = WarmUpLr(WARMUP_STEPS, D_MODEL, optim.param_groups)
    translator.to(DEVICE)
    min_valid_loss = 1e10
    epochs = 0
    it = 0.
    translator.eval()
    train_sentence = EN_TRAIN[30]
    valid_sentence = EN_VALID[40]

    while True:
        print('Start epoch %d' % epochs)
        translator.train()
        for (en_sentences, en_lengths),  (ger_sentences, ger_lengths) in trainloader:
            start = time()
            en_sentences, ger_sentences = en_sentences.to(DEVICE), ger_sentences.to(DEVICE)
            en_lengths, ger_lengths = en_lengths.to(DEVICE), ger_lengths.to(DEVICE)
            optim.zero_grad()
            output = translator.forward(en_sentences, ger_sentences, en_lengths, ger_lengths)
            loss_item = loss_function(output, ger_sentences)
            loss_item.backward()
            optim.step()
            end = time()
            print(f'TRAIN iteration {it}; loss: {loss_item.item()}; '
                  f'lr: {optim.param_groups[0]["lr"]}; iteration time: {end - start}')

        if it % 500 == 0:
            with torch.no_grad():
                translator.eval()
                total_loss = 0.
                eval_it = 0.
                for (en_sentences, en_lengths), (ger_sentences, ger_lengths) in validloader:

                    if eval_it == 0:
                        print('\n' + train_sentence + '\n')
                        print(translator.predict(train_sentence), '\n')

                        print(valid_sentence + '\n')
                        print(translator.predict(valid_sentence), '\n')

                    en_sentences, ger_sentences = en_sentences.to(DEVICE), ger_sentences.to(DEVICE)
                    en_lengths, ger_lengths = en_lengths.to(DEVICE), ger_lengths.to(DEVICE)
                    output = translator.forward(en_sentences, ger_sentences, en_lengths, ger_lengths)
                    loss_item = loss_function(output, ger_sentences)
                    total_loss += loss_item.item()
                    eval_it += 1
                total_loss = total_loss / eval_it
                print(f'VALID iteration {it}; loss: {total_loss};')
                if total_loss < min_valid_loss:
                    min_valid_loss = total_loss
                    print('Saving...')
                    checkpoint = {
                        'n': N, 'vocab_size': VOCAB_SIZE, 'seq_len': SEQ_LEN,
                        'd_model': D_MODEL, 'd_k': D_K, 'd_v': D_V, 'h': H,
                        'd_ff': D_FF, 'state_dict': translator.state_dict()
                    }
                    torch.save(checkpoint, CHECKPOINT_PATH)
            translator.train()
        it += 1
        lr_schedule(it)
    epochs += 1


if __name__ == '__main__':
    train()
