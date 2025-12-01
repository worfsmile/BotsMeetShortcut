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

def _test(model, test_loader, device):
    model.eval()
    y_pred, y_true = [], []
    probs = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model([data])
            prob = nn.functional.softmax(out, dim=1)
            probs.extend(prob.tolist())
            out = torch.argmax(out, dim=1).tolist()
            y_pred.extend(out)
            y_true.extend(label.tolist())

    acc_test = accuracy_score(y_pred, y_true)
    probs = np.array(probs)
    y_true = np.array(y_true)
    ece_test = ece_metric.measure(probs, y_true)
    return acc_test, ece_test

def run(models, loaders, device, logname):

    seeds = [114514]
    epochs = 500
    early_stop = 100
    
    train_loader, test_loader, val_loader = loaders
    model, optimizer, loss_fn = models

    for seed in seeds:
        set_seed(seed)
        best_valid = {'acc': 0}
        best_test = {'acc': 0, 'ece': 0}
        early_stop_count = 0

        print('Start training...')
        for epoch in range(epochs):
            model.train()
            epoch_loss, correct, total = 0, 0, 0

            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                output = model([data])
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total

            valid_acc, _ = _test(model, val_loader, device)
            test_acc, test_ece = _test(model, test_loader, device)

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
