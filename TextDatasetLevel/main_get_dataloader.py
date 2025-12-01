from train import run
import torch
from sklearn.model_selection import train_test_split
from model import MLP
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import random
import json

data_dir = "."


with open('./data/basic/id2label.json', 'r') as f:
    label2id = json.load(f)

id2label = {}
for key in label2id:
    id2label[key] = {}
    for l in label2id[key]:
        id2label[key][label2id[key][l]] = l

with open('./data/split_idx/config_trans.json', 'r') as f:
    split_idx = json.load(f)

def split_data(index, test_size=0.1, random_state=42):
    train_index, test_index = train_test_split(index, test_size=test_size, random_state=random_state)
    train_index = sorted(train_index)
    test_index = sorted(test_index)
    return torch.tensor(train_index), torch.tensor(test_index)

def get_llm_dataloader(dataset, text, feature, train_type, mod_flag = None):
    idx_dir = f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/{train_type}"
    train_idx = torch.load(f"{idx_dir}/train_idx.pt")
    test_idx = torch.load(f"{idx_dir}/test_idx.pt")
    label = torch.load(f"{data_dir}/data/ood2/{dataset}/label.pt")

    llm_idx_dir = f"{data_dir}/llm_enhance/{dataset}/{feature}/{text}"
    des_tensor = torch.load(f"{llm_idx_dir}/{mod_flag}/tensors.pt")

    if os.path.exists(f"{llm_idx_dir}/{mod_flag}/idxs"):
        train_idx = torch.load(f"{llm_idx_dir}/{mod_flag}/idxs/train_idx.pt")
        val_idx = torch.load(f"{llm_idx_dir}/{mod_flag}/idxs/val_idx.pt")
    else:
        os.makedirs(f"{llm_idx_dir}/{mod_flag}/idxs", exist_ok=True)
        # 找到des_tensor中行全0的索引
        all_zero_mask = (des_tensor == 0).all(dim=1)  # shape (N,)
        zero_idx = all_zero_mask.nonzero(as_tuple=False).squeeze(1)  # shape (K,)
        # train_idx中不包含zero_idx
        train_idx = train_idx.tolist()
        zero_idx = zero_idx.tolist()
        train_idx = list(set(train_idx) - set(zero_idx))
        train_idx = torch.tensor(train_idx)
        print(f"train_idx: {len(train_idx)}")
        # 划分train_idx和val_idx
        train_idx, val_idx = split_data(train_idx, test_size=0.1, random_state=42)
        torch.save(train_idx, f"{llm_idx_dir}/{mod_flag}/idxs/train_idx.pt")
        torch.save(val_idx, f"{llm_idx_dir}/{mod_flag}/idxs/val_idx.pt")


    train_des = des_tensor[train_idx]
    train_label = label[train_idx]
    val_des = des_tensor[val_idx]
    val_label = label[val_idx]

    des_tensor = torch.load(f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/tensors.pt")

    test_des = des_tensor[test_idx]
    test_label = label[test_idx]

    train_loader = DataLoader(TensorDataset(train_des, train_label), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_des, val_label), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_des, test_label), batch_size=64, shuffle=False)
    return train_loader, test_loader, val_loader, des_tensor.shape[1]

def get_ood_dataloader(dataset, text, feature, train_type):
    idx_dir = f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/{train_type}"
    train_idx = torch.load(f"{idx_dir}/train_idx.pt")
    test_idx = torch.load(f"{idx_dir}/test_idx.pt")

    des_tensor = torch.load(f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/tensors.pt")
    
    label = torch.load(f"{data_dir}/data/ood2/{dataset}/label.pt")
    if os.path.exists(f"{idx_dir}/val_idx.pt"):
        val_idx = torch.load(f"{idx_dir}/val_idx.pt").tolist()
        train_idx = train_idx.tolist()
        train_idx = list(set(train_idx) - set(val_idx))
        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)
    else:
        train_idx, val_idx = split_data(train_idx, test_size=0.1, random_state=42)
        torch.save(val_idx, f"{idx_dir}/val_idx.pt")
    
    if text == "tweets":
        positive_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/positive_set.pt").tolist()
        negative_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/negative_set.pt").tolist()
    
        check_idx = positive_set + negative_set

        if not set(train_idx.tolist()).issubset(set(check_idx)):
            raise ValueError("train_idx is not subset of positive_set and negative_set")

    train_des = des_tensor[train_idx]

    all_zero_rows = (train_des == 0).all(dim=1)

    if all_zero_rows.any():
        zero_row_indices = all_zero_rows.nonzero(as_tuple=False).squeeze(1)
        raise ValueError(f"train_des has all-zero rows len: {len(zero_row_indices)}")

    train_label = label[train_idx]
    val_des = des_tensor[val_idx]
    val_label = label[val_idx]
    test_des = des_tensor[test_idx]
    test_label = label[test_idx]

    train_loader = DataLoader(TensorDataset(train_des, train_label), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_des, val_label), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_des, test_label), batch_size=64, shuffle=False)
    return train_loader, test_loader, val_loader, des_tensor.shape[1]


def get_dataloader(dataset, text, feature, train_type, mod_tag, mod_flag):
    if mod_tag == "ood":
        return get_ood_dataloader(dataset, text, feature, train_type)
    elif mod_tag == "llm":
        return get_llm_dataloader(dataset, text, feature, train_type, mod_flag)
    else:
        raise ValueError("train_type should be ood or llm")
