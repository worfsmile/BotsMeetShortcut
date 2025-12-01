# Convert the generated macro-AMRs to subgraphs in DGL format using

import glob
import networkx as nx
import penman
import amrlib
import pandas as pd
import penman
from penman import constant
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.alignments.rbw_aligner import RBWAligner
from penman.models.noop import NoOpModel
import ast
import pickle
import os
# import dgl
import json
import numpy as np
from sklearn.model_selection import train_test_split
import logging
logging.getLogger('penman').setLevel(logging.ERROR)
from torch_geometric.utils import from_networkx
import argparse
import torch

def var2word(p_graph):
    v2w = {}
    for (source, _, target) in p_graph.instances():
        v2w[source] = target
    return v2w

#**************Specify glove embedding path*************
def get_glove():
    glove = {}
    f = open(f'{GLOVE_EMBEDDING_PATH}', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove[word] = coefs
    return glove

def to_dict(d, EMBEDDING_DIM):
    tmp = {}
    for i, j in d.items():
        if isinstance(j, list):
            j = np.array(j)
        if j.shape[0] < EMBEDDING_DIM:
            j = np.pad(j, (0, EMBEDDING_DIM-j.shape[0]), 'constant', constant_values=0)
        elif j.shape[0] > EMBEDDING_DIM:
            j = j[:EMBEDDING_DIM]
        tmp[i] = {'feat':j}
    return tmp

def id2label(df):
    return dict(zip(df['u_id'], zip(df['label'], df['tweets'])))

# def convert_to_pyg_data(item):
#     dgl_graph = item['graph']
#     x = dgl_graph.ndata['feat'] if 'feat' in dgl_graph.ndata else None
#     if x is None:
#         x = torch.ones(dgl_graph.num_nodes, 1)
#     x = torch.tensor(x, dtype=torch.float)
#     src, dst = dgl_graph.edges()
#     edge_index = torch.stack([src, dst], dim=0)
#     edge_attr = dgl_graph.edata['feat'] if 'feat' in dgl_graph.edata else None
#     if edge_attr is None:
#         edge_attr = torch.ones(edge_index.shape[1], 1)
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
#     graph_label = item.get('label', None)
#     if graph_label is not None:
#         try:
#             graph_label = int(graph_label)  # 将字符串 '0' 转换为整数 0
#         except:
#             if graph_label == 'bot' or graph_label == '1':
#                 graph_label = 1
#             elif graph_label == 'human' or graph_label == '0':
#                 graph_label = 0
#         graph_label = torch.tensor([graph_label])  # 将标签转为张量格式
#         graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=graph_label)
#     return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type = str, default='back5', help = "Specify the number of text to be used for each AMR.")
    parser.add_argument('--feature', type = str, default='sentiments', help = "Specify the feature to be used for each AMR.")
    parser.add_argument('--dataset', default = 'twi20', choices = ['csi15', 'csi17', 'mid18', 'twi20'], help='Specify the dataset for which you want to run the experiments.')

    # Parse the argument
    args = parser.parse_args()

    GLOVE_EMBEDDING_PATH = "./8AMR_CIGA/t-ai9/glove.6B.50d.txt"

    EMBEDDING_DIM = 50
    
    datasets = ["cresci-2015-data", "cresci-2017-data", "cresci-stock-2018", "midterm-2018", "twibot-20"]
    texts = ['description', 'tweets']
    features = ['sentiments', 'emotions', 'topics', 'values']
    

    for data_name in datasets:
        for text in texts:
            for feature in features:
                args.text = text
                args.feature = feature
                
                print(f"Processing {data_name} dataset...")
                args.dataset = data_name
                glove = get_glove()
                raw_json_path = f'./deal_dataset/{args.dataset}/u_tweets_split_feature.json'
                with open(raw_json_path, 'r', encoding='utf-8') as f:
                    raw_json = json.load(f)
                
                raw_dict_id2idx_path = f'./deal_dataset/{args.dataset}/id2idx.json'
                
                with open(raw_dict_id2idx_path, 'r', encoding='utf-8') as f:
                    raw_dict_id2idx = json.load(f)

                merged_amr= glob.glob(f"./8AMR_CIGA/data/amr_merge/{args.dataset}/{args.text}/{args.feature}/*.amr.penman")
                dataset = []
                lv = 0
                for curr in merged_amr:
                    p_graph = penman.load(curr, model = NoOpModel())[0]
                    name = os.path.splitext(os.path.basename(curr))[0]
                    name = name.split('.')[0]
                    name = name.split('_')[0]
                    idx = int(raw_dict_id2idx[name])

                    # if name not in i2l:
                    #     continue
                    v2w = var2word(p_graph)
                    nx_graph = nx.MultiDiGraph()
                    nx_graph.add_edges_from([(s, t) for s, _, t in p_graph.edges()])#TODO: Add edges from instances as well

                    #-----------------------------------extracting subgraphs----------------------------------------------------
                    #sorted ordering is a must in order to preserve the node order in case of using from_networkx
                    temp= nx.convert_node_labels_to_integers(nx_graph, ordering = 'sorted', label_attribute= 'original')
                    original2new  = {temp.nodes[i]['original']:i for i in temp.nodes}
                    subgraphs = [[ original2new[j] for j in i] for i in eval(p_graph.metadata['subgraphs'])]
                    #-----------------------------------------------------------------------------------------------------------

                    MAP = {i:glove.get(v2w[i], [0]*EMBEDDING_DIM) for i in nx_graph.nodes()}
                    attr= to_dict(MAP, EMBEDDING_DIM)
                    nx.set_node_attributes(nx_graph, attr)

                    pyg_graph = from_networkx(nx_graph)
                    
                    label = raw_json[name]['label']
                    sample = {'label':label, 'graph': pyg_graph, 'id': name, 'subgraphs':subgraphs, 'idx':idx}
                    dataset.append(sample)
                    lv+=1
                    print("done", lv)
                save_path = f"./8AMR_CIGA/data/data_pyg/{args.dataset}/{args.text}/{args.feature}/data.pkl"
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(dataset, f)

                