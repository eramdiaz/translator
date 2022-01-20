"""Train the Transformer"""

import os
import pandas as pd
import torch
from uuid import uuid4
from torch.utils.data import DataLoader
from translator.data import ItemGetter
from translator.models import Transformer
from translator.learning_rate import WarmUpLr


N = 2
D_MODEL = 512
SEQ_LEN = 100
H = 8
D_K = D_V = 64
D_FF = 2048
VOCAB_SIZE = 8000
WARMUP_STEPS = 4000
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_FOLDER = '/'.join(__file__.split('/', -2)[:-2]) + '/checkpoints'
assert os.path.exists(CHECKPOINTS_FOLDER), \
    'Create a checkpoints folder in translator for saving the model.'
CHECKPOINT = CHECKPOINTS_FOLDER + '/' + str(uuid4()) + '.pth'

translator = Transformer(N, VOCAB_SIZE, SEQ_LEN, D_MODEL, D_K, D_V, H, D_FF)


EN_TRAIN = pd.read_csv('data/en_train.txt', delimiter='\n', header=None)[0].tolist() #nrows
DE_TRAIN = pd.read_csv('data/de_train.txt', delimiter='\n', header=None)[0].tolist()
EN_VALID = pd.read_csv('data/en_valid.txt', delimiter='\n', header=None)[0].tolist()
DE_VALID = pd.read_csv('data/en_valid.txt', delimiter='\n', header=None)[0].tolist()

train_dataset = ItemGetter(EN_TRAIN, DE_TRAIN, SEQ_LEN)
valid_dataset = ItemGetter(EN_VALID, DE_VALID, SEQ_LEN)

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

loss = torch.nn.CrosEntropyLoss(ignore_index=-10)


def loss_function(output, targets):
    sentences, lengths = targets
    total_loss = 0.
    for batch, sentence, length in zip(output, sentences, lengths):
        true_sentence = sentence[1:length.item()]
        loss_item = loss(batch[:(length.item() - 1), :], true_sentence)
        total_loss += loss_item
    return total_loss


def train():
    optim = torch.optim.Adam(translator.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-09)
    lr_schedule = WarmUpLr(WARMUP_STEPS, D_MODEL, optim.param_groups)
    translator.to(DEVICE)
    min_valid_loss = 1e10
    epochs = 0
    it = 0.
#    test_sentence_1 = train_set[30][0]
#    test_sentence_2 = valid_set[40][0]
#
#    print(test_sentence_1 + '\n')
#    print(train_set[30][1])
#    print('\n')
#    prediction = predict_function(test_sentence_1)
#    print(bpe._model.decode(prediction) + '\n')
#    print(test_sentence_2 + '\n')
#    print(valid_set[40][1])
#    print('\n')
#    prediction = predict_function(test_sentence_2)
#    print(bpe._model.decode(prediction) + '\n')

    while True:
        print('Start epoch %d' % epochs)
        translator.train()
        for eng_sentences, ger_sentences in trainloader:
            eng_sentences, ger_sentences = eng_sentences.to(DEVICE), ger_sentences.to(DEVICE)
            optim.zero_grad()
            output = translator.forward(eng_sentences, ger_sentences)
            loss_item = loss_function(output, ger_sentences)
            loss_item.backward()
            optim.step()
            print(f'TRAIN iteration {it}; loss: {loss_item.item()}; lr: {optim.param_groups[0]["lr"]}')

        if it % 500 == 0:
            with torch.no_grad():
                translator.eval()
                total_loss = 0.
                eval_it = 0.
                for eng_sentences, ger_sentences in validloader:
#                    if eval_it == 0:
#                        print('\n' + test_sentence_1 + '\n')
#                        prediction = predict_function(test_sentence_1)
#                        print(bpe._model.decode(prediction) + '\n')
#
#                        print(test_sentence_2 + '\n')
#                        prediction = predict_function(test_sentence_2)
#                        print(bpe._model.decode(prediction) + '\n')

                    output = translator.forward(eng_sentences, ger_sentences)
                    loss_item = loss_function(output, ger_sentences)
                    total_loss += loss_item.item()
                    eval_it += 1
                total_loss = total_loss / eval_it
                print(f'VALID iteration {it}; loss: {total_loss};')
                if total_loss < min_valid_loss:
                    min_valid_loss = total_loss
                    print('Saving...')
                    torch.save(translator.state_dict(), CHECKPOINT)
            translator.train()
        it += 1
        lr_schedule(it)
    epochs += 1


if __name__ == '__main__':
    train()
