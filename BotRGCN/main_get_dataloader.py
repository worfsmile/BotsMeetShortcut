from train import run
import torch
from sklearn.model_selection import train_test_split
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
    edge_index = torch.load(f"{data_dir}/data/ood2/{dataset}/edge_index.pt")
    edge_type = torch.load(f"{data_dir}/data/ood2/{dataset}/edge_type.pt")

    llm_idx_dir = f"{data_dir}/llm_enhance/{dataset}/{feature}/{text}"
    des_tensor = torch.load(f"{llm_idx_dir}/{mod_flag}/tensors.pt")
    raw_tensor = torch.load(f"{data_dir}/data/ood2/{dataset}/{text}/tensors.pt")

    if os.path.exists(f"{llm_idx_dir}/{mod_flag}/idxs"):
        train_idx = torch.load(f"{llm_idx_dir}/{mod_flag}/idxs/train_idx.pt")
        val_idx = torch.load(f"{llm_idx_dir}/{mod_flag}/idxs/val_idx.pt")
    else:
        os.makedirs(f"{llm_idx_dir}/{mod_flag}/idxs", exist_ok=True)
        all_zero_mask = (des_tensor == 0).all(dim=1)
        zero_idx = all_zero_mask.nonzero(as_tuple=False).squeeze(1)
        train_idx = train_idx.tolist()
        zero_idx = zero_idx.tolist()
        train_idx = list(set(train_idx) - set(zero_idx))
        train_idx = torch.tensor(train_idx)
        print(f"train_idx: {len(train_idx)}")
        train_idx, val_idx = split_data(train_idx, test_size=0.1, random_state=42)
        torch.save(train_idx, f"{llm_idx_dir}/{mod_flag}/idxs/train_idx.pt")
        torch.save(val_idx, f"{llm_idx_dir}/{mod_flag}/idxs/val_idx.pt")

    all_zero_mask = (des_tensor == 0).all(dim=1)
    zero_idx = all_zero_mask.nonzero(as_tuple=False).squeeze(1)
    train_idx = train_idx.tolist()
    zero_idx = zero_idx.tolist()
    for i in range(des_tensor.shape[0]):
        if i in zero_idx:
            continue
        raw_tensor[i] = des_tensor[i]

    return train_idx, val_idx, test_idx, raw_tensor, label, edge_index, edge_type

def get_ood_dataloader(dataset, text, feature, train_type):
    idx_dir = f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/{train_type}"
    train_idx = torch.load(f"{idx_dir}/train_idx.pt")
    test_idx = torch.load(f"{idx_dir}/test_idx.pt")
    
    des_tensor = torch.load(f"{data_dir}/data/ood2/{dataset}/{text}/{feature}/tensors.pt")
    raw_tensor = torch.load(f"{data_dir}/data/ood2/{dataset}/{text}/tensors.pt")

    label = torch.load(f"{data_dir}/data/ood2/{dataset}/label.pt")
    edge_index = torch.load(f"{data_dir}/data/ood2/{dataset}/edge_index.pt")
    edge_type = torch.load(f"{data_dir}/data/ood2/{dataset}/edge_type.pt")

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
        positive_set = torch.load(f"{data_dir}/data/ood2/{dataset}/tweets/{feature}/positive_set.pt").tolist()
        negative_set = torch.load(f"{data_dir}/data/ood2/{dataset}/tweets/{feature}/negative_set.pt").tolist()
    
        check_idx = positive_set + negative_set

        if not set(train_idx.tolist()).issubset(set(check_idx)):
            raise ValueError("train_idx is not subset of positive_set and negative_set")

    all_zero_mask = (des_tensor == 0).all(dim=1)
    zero_idx = all_zero_mask.nonzero(as_tuple=False).squeeze(1)
    train_idx = train_idx.tolist()
    zero_idx = zero_idx.tolist()
    for i in range(des_tensor.shape[0]):
        if i in zero_idx:
            continue
        raw_tensor[i] = des_tensor[i]

    return train_idx, val_idx, test_idx, raw_tensor, label, edge_index, edge_type


def get_dataloader(dataset, text, feature, train_type, mod_tag, mod_flag):
    if mod_tag == "ood":
        return get_ood_dataloader(dataset, text, feature, train_type)
    elif mod_tag == "llm":
        return get_llm_dataloader(dataset, text, feature, train_type, mod_flag)
    else:
        raise ValueError("train_type should be ood or llm")
