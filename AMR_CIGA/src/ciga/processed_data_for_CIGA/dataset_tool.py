
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data_dir = './8AMR_CIGA/data'

class CustomPygDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        super().__init__(None, transform)
        self.data, self.slices = self.collate(data_list)

    def get(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

def convert_to_pyg_data(item):
    dgl_graph = item['graph']
    x = dgl_graph.feat
    if x is None:
        x = torch.ones(dgl_graph.num_nodes, 1)
    x = torch.tensor(x, dtype=torch.float)
    src, dst = dgl_graph.edge_index
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = dgl_graph.edge_attr if getattr(dgl_graph, 'edge_attr', None) is not None else None
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.shape[1], 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    graph_label = item.get('label', None)
    if graph_label is not None:
        try:
            graph_label = int(graph_label)
        except:
            if graph_label == 'bot' or graph_label == '1':
                graph_label = 1
            elif graph_label == 'human' or graph_label == '0':
                graph_label = 0
        graph_label = torch.tensor([graph_label])
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=graph_label)
    return graph
