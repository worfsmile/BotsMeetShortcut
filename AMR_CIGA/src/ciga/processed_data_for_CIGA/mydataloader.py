
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
from .normal_dataloader import pkl_to_pyg_dataset

def get_dataloader(args):
    dataset, input_dim, num_classes, edge_dim, train_indices, valid_indices, test_indices = pkl_to_pyg_dataset(args)
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
   
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    split_idx = {
    "train": train_indices,
    "valid": valid_indices,
    "test": test_indices
    }

    return train_loader, valid_loader, test_loader, input_dim, num_classes, edge_dim

if __name__ == '__main__':
    class Args:
        dataset = 'csi17'
        train_ratio = 0.2
        valid_ratio = 0.5
        test_ratio = 0.8
    args = Args()
    train_dataset, valid_dataset, test_dataset, input_dim, num_classes, edge_dim = get_dataloader(args)
    print("train_dataset:", len(train_dataset))
    print("valid_dataset:", len(valid_dataset))
    print("test_dataset:", len(test_dataset))
    print("input_dim:", input_dim)
    print("num_classes:", num_classes)
    print("edge_dim:", edge_dim)