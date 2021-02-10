from src.modulos.dataset import Uti
from src.modulos.rede import ModeloNeural

from torch.utils.data import DataLoader
import torch
from torch import optim, nn

import numpy as np

from sklearn.metrics import accuracy_score

import time

# Argumentos da rede
args = dict(
    batch_size=50,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    weight_decay = 1e1,
    lr = 1e4,
    epoch = 100
)

# Instancia dos dados
train_set = Uti('dados/x_train.csv', 'dados/y_train.csv')
test_set = Uti('dados/x_test.csv', 'dados/y_test.csv')

train_loader = DataLoader(train_set, args['batch_size'])
test_loader = DataLoader(test_set, args['batch_size'])

# Rede neural
net = ModeloNeural(train_set.quantidade_features, 500, 300, 2)
net = net.to(args['device'])

# Otimizador e perda
otimizador = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
perda = nn.CrossEntropyLoss().to(args['device'])


def train(modelo: ModeloNeural, train_loader: DataLoader, epoch : int):
    start = time.time()
    modelo.train()

    epoch_loss = []
    pred_list, rotulo_list = [], []
    for batch in train_loader:
        dado, rotulo = batch

        # cast para gpu
        dado = dado.to(args['device'])
        rotulo = rotulo.to(args['device']).view(-1)

        # forward
        predicao = modelo(dado)
        loss = perda(predicao, rotulo)
        epoch_loss.append(loss.cpu().data)

        _, pred = torch.max(predicao, axis =1)*
        pred_list.append(predicao.cpu().numpy())
        rotulo_list.append(rotulo.cpu().numpy())

        # Backward
        otimizador.zero_grad()
        loss.backward()
        otimizador.step()

    epoch_loss = np.asarray(epoch_loss)
    pred_list = np.asarray(pred_list).ravel()
    rotulo_list = np.asarray(rotulo_list).ravel()

    acc = accuracy_score(pred_list, rotulo_list) * 100

    end = time.time()

    print(f'Época {epoch} - loss {epoch_loss.mean():0.2f} +/-{epoch_loss.std():0.2f} - Acc{acc:0.2f} '
          f'Tempo:{end-start:0.2f}', end=' ')

    return epoch_loss.mean(), acc


def validacao(modelo, test_loader:DataLoader):
    start = time.time()
    modelo.eval()

    epoch_loss = []
    pred_list, rotulo_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            dado, rotulo = batch

            # cast para gpu
            dado = dado.to(args['device'])
            rotulo = rotulo.to(args['device']).view(-1)

            # forward
            predicao = modelo(dado)
            loss = perda(predicao, rotulo)
            epoch_loss.append(loss.cpu().data)

            _, pred = torch.max(predicao, axis =1)
            pred_list.append(pred.cpu().numpy())
            rotulo_list.append(rotulo.cpu().numpy())

    epoch_loss = np.asarray(epoch_loss)
    pred_list = np.asarray(pred_list).ravel()
    rotulo_list = np.asarray(rotulo_list).ravel()

    acc = accuracy_score(pred_list, rotulo_list) * 100

    end = time.time()

    print(f'- val_loss {epoch_loss.mean() : 0.2f} - Acc {acc:0.2f} - Tempo {end - start : 0.2f}')

    return epoch_loss.mean(), acc


train_losses, test_losses = [], []
for epoch in range(args['epoch']):
    # Train
    train_losses.append(train(net, train_loader, epoch))

    # Validate
    test_losses.append(validacao(net, test_loader))