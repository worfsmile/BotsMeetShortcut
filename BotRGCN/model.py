from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv


def _linear_layer(input_size: int, output_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LeakyReLU()
    )

class BotRGCN(nn.Module):
    def __init__(self,
        input_dims: List[int] = [768, 768],
        embedding_dim: int = 64,
        dropout: float = 0.3
    ):
        super(BotRGCN, self).__init__()
        self.dropout = dropout

        l = len(input_dims)
        self.input_layers = nn.ModuleList([
            _linear_layer(_in, embedding_dim//l)
            for _in in input_dims
        ])
        self.fc1 = _linear_layer(embedding_dim//l*l, embedding_dim)
        self.rgcn1=RGCNConv(
            embedding_dim,
            embedding_dim,
            num_relations=2
        )
        self.rgcn2=RGCNConv(
            embedding_dim,
            embedding_dim,
            num_relations=2
        )
        self.output = nn.Sequential(
            _linear_layer(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim,2)
        )

    def forward(self,
        inputs: List[torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ):
        out = []
        for layer, _input in zip(self.input_layers, inputs):
            out.append(layer(_input))

        x = torch.cat(out, dim=1)
        x = self.fc1(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        return self.output(x)

