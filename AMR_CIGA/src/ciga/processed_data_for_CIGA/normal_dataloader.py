
import pickle
from sklearn.model_selection import train_test_split
from .dataset_tool import *
import torch
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def pkl_to_pyg_dataset(args):
    file_path = f'./8AMR_CIGA/data/data_pyg/{args.dataset}/{args.text}/{args.feature}/data.pkl'
    train_idx_path = f'./data/ood2/{args.dataset}/{args.text}/{args.feature}/{args.ood_type}/train_idx.pt'
    val_idx_path = f'./8AMR_CIGA/data/ood2/{args.dataset}/{args.text}/{args.feature}/{args.ood_type}/val_idx.pt'
    test_idx_path = f'./data/ood2/{args.dataset}/{args.text}/{args.feature}/{args.ood_type}/test_idx.pt'

    train_idx = torch.load(train_idx_path)
    val_idx = torch.load(val_idx_path)
    test_idx = torch.load(test_idx_path)

    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)
    
    pyg_data_list = []
    for item in data_list:
        pyg_data_list.append(convert_to_pyg_data(item)) 
    dataset = CustomPygDataset(pyg_data_list)
    input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else None
    all_labels = [data.y.item() for data in dataset if data.y is not None]
    num_classes = len(set(all_labels))
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else None
    
    # valid_ratio = args.valid_ratio

    num_graphs = len(dataset)
    indices = list(range(num_graphs))

    train_indices = []
    for i in range(len(data_list)):
        if data_list[i]['idx'] in train_idx:
            train_indices.append(i)
    test_indices = []
    for i in range(len(data_list)):
        if data_list[i]['idx'] in test_idx:
            test_indices.append(i)
    valid_indices = []
    for i in range(len(data_list)):
        if data_list[i]['idx'] in val_idx:
            valid_indices.append(i)

    # train_indices, valid_indices = train_test_split(train_indices, test_size=valid_ratio, shuffle=True, random_state=42)
    
    # print(f"Input dimension (node feature dimension): {input_dim}")
    # print(f"Number of classes: {num_classes}")
    # print(f"Edge dimension: {edge_dim}")
    # print(dataset)
    # print("Number of graphs:", len(dataset))
    # print("Example graph:", dataset[0])

    return dataset, input_dim, num_classes, edge_dim, train_indices, valid_indices, test_indices