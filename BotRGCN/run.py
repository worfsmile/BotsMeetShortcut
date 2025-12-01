from train import run
import torch
from sklearn.model_selection import train_test_split
from model import BotRGCN
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from main_get_dataloader import get_dataloader
import json
import random

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(114514)

def run_one_dataset(dataset, text, feature, train_type, device, global_table, ece_table):
    train_idx, val_idx, test_idx, features, labels, edge_index, edge_type = get_dataloader(dataset, text, feature, train_type, mod_tag, mod_flag)
    input_size = features.shape[1]
    model = BotRGCN([input_size]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    logname = f"{dataset}_{text}_{feature}_{train_type}"
    model, best_acc, ece = run((model, optimizer, criterion), (train_idx, val_idx, test_idx, features, labels, edge_index, edge_type), device, logname)
    global_table[dataset][text][feature][train_type] = best_acc
    ece_table[dataset][text][feature][train_type] = ece
    return global_table, ece_table

train_setting = 'ood2'
data_dir = "."

huge_table = {}
for train_setting in ['llm_enhance1', 'llm_enhance3', "ood2"]:
    if train_setting == 'llm_enhance1':
        train_types = ['00', '01']
        mod_flag = 'select_idx'
        mod_tag = 'llm'

    if train_setting == 'llm_enhance3':
        train_types = ['00', '01']
        mod_flag = 'mix_idx'
        mod_tag = 'llm'

    if train_setting == 'ood2':
        train_types = ['00', '01', '10', '11']
        mod_tag = 'ood'
        mod_flag = None

    datasets = ['cresci-2015-data', 'twibot-20']
    texts = ["description", 'tweets']
    features = ["sentiments", "topics", "values", "emotions"]

    global_table = {}
    distribution = {}
    ece_table = {}
    for dataset in datasets:
        global_table[dataset] = {}
        distribution[dataset] = {}
        ece_table[dataset] = {}
        for text in texts:
            global_table[dataset][text] = {}
            distribution[dataset][text] = {}
            ece_table[dataset][text] = {}
            for feature in features:
                global_table[dataset][text][feature] = {}
                distribution[dataset][text][feature] = {}
                ece_table[dataset][text][feature] = {}
                for train_type in train_types:
                    if not os.path.exists(f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/{train_type}"):
                            continue
                    if text == 'description' and train_setting == 'llm_enhance3':
                        continue
                    print(dataset, text, feature, train_type, train_setting)
                    global_table, ece_table = run_one_dataset(dataset, text, feature, train_type, "cuda:0", global_table, ece_table)

    huge_table[train_setting] = global_table

arrange_table = {}

for ds in datasets:
    arrange_table[ds] = {}
    for i in [0, 1]:
        i = str(i)
        arrange_table[ds][i] = {}
        for j in ['standard', 'shortcut']:
            arrange_table[ds][i][j] = {}
            for f in features:
                arrange_table[ds][i][j][f] = {}
                for t in texts:
                    if j == 'standard':
                        _j = 1
                    if j == 'shortcut':
                        _j = 0
                    train_type = f'{_j}{i}'
                    try:
                        arrange_table[ds][i][j][f][t] = huge_table['ood2'][ds][t][f][train_type]
                    except:
                        arrange_table[ds][i][j][f][t] = -1

for ds in datasets:
    for i in [0, 1]:
        i = str(i)
        for j in ['1', '2', '3']:
            aug = f"augmentation{j}"
            arrange_table[ds][i][aug] = {}
            for f in features:
                arrange_table[ds][i][aug][f] = {}
                for t in texts:
                    _j = 0
                    train_type = f'{_j}{i}'
                    try:
                        arrange_table[ds][i][aug][f][t] = huge_table[f'llm_enhance{j}'][ds][t][f][train_type]
                    except:
                        arrange_table[ds][i][aug][f][t] = -1


