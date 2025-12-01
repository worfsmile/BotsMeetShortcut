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

class MLP(nn.Module):
    def __init__(self,
        input_dims: List[int] = [768, 768], 
        embedding_dim: int = 128
    ):
        super(MLP, self).__init__()

        self.input_layers = nn.ModuleList([
            _linear_layer(_in, embedding_dim//len(input_dims))
            for _in in input_dims
        ])
        self.output = nn.Sequential(
            _linear_layer(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, 2)
        )

    def forward(self, inputs: List[torch.Tensor]):
        out = []
        for layer, _input in zip(self.input_layers, inputs):
            out.append(layer(_input))
        return self.output(torch.cat(out, dim=1))