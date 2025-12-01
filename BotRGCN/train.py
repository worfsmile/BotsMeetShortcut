import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score
from netcal.metrics import ECE
import wandb

ece_metric = ECE(bins=10)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(114514)

def _test(output, labels, test_idx):
    y_pred = output[test_idx].argmax(dim=1).tolist()
    y_true = labels[test_idx].tolist()
    probs = output[test_idx].softmax(dim=1).tolist()
    acc_test = accuracy_score(y_pred, y_true)
    probs = np.array(probs)
    y_true = np.array(y_true)
    ece_test = ece_metric.measure(probs, y_true)
    return acc_test, ece_test

def run(models, loaders, device, logname):

    seeds = [114514]
    epochs = 500
    early_stop = 100
    
    train_idx, val_idx, test_idx, features, labels, edge_index, edge_type = loaders
    
    features = features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    assert features.shape[0] >= edge_index.max().item() + 1, f'Too many nodes in the graph. {features.shape[0]} < {edge_index.max().item() + 1}'

    model, optimizer, loss_fn = models

    for seed in seeds:
        set_seed(seed)
        best_valid = {'acc': 0}
        best_test = {'acc': 0, 'ece': 0}
        early_stop_count = 0

        print('Start training...')
        for epoch in range(epochs):
            model.train()

            output = model([features], edge_index, edge_type)
            loss = loss_fn(output[train_idx], labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = accuracy_score(output[train_idx].argmax(dim=1).tolist(), labels[train_idx].tolist())
            train_loss = loss.item()

            model.eval()    
            valid_acc, valid_ece = _test(output, labels,  val_idx)
            test_acc, test_ece = _test(output, labels,  test_idx)

            if valid_acc >= best_valid['acc']:
                if valid_acc > best_valid['acc']:
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                best_valid.update({'acc': valid_acc})
                best_test.update({'acc': test_acc})
                best_test.update({'ece': test_ece})
 
            else:
                early_stop_count += 1
            if early_stop_count > early_stop:
                print(f'Early stop at epoch {epoch}')
                break

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\nValid Acc\Best val: {valid_acc:.4f}\{best_valid["acc"]:.4f}, Test Acc\Best Test: {test_acc:.4f}\{best_test["acc"]:.4f}, Test ECE\Best Test: {test_ece:.4f}\{best_test["ece"]:.4f}')

    print(f'Best Test Acc: {best_test["acc"]:.4f}')

    return  model, best_test["acc"], best_test["ece"]
