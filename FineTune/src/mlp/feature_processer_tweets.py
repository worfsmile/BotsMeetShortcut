
import os
import json
import torch
import multiprocessing as mp

import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
import sys

import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, pipeline
import torch
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import argparse
import traceback
from multiprocessing import Process, cpu_count
from transformers.utils import logging

logging.set_verbosity_error()

def deal(data, cuda_th, fro, to, data_dir, dataset, feature):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_th)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    each_user_tweets = data[fro:to]
    feature_extract = pipeline(
        'feature-extraction',
        model='./6FineTune/output/fine_tune_model/best_model',
        # model = '',
        tokenizer='roberta-base',
        device=device,
        padding=True, 
        truncation=True, 
        max_length=512,
        add_special_tokens=True
    )

    text_features = []
    count = 0

    for tweets in tqdm(each_user_tweets):
        if len(" ".join(tweets).strip()) == 0:
            text_features.append(torch.zeros(768))
            count += 1
            continue
        
        num_tweets = len(tweets)
        total_each_person_tweets = torch.zeros(768)
        
        for j in range(num_tweets):
            each_tweet = tweets[j]
            
            if not each_tweet:
                total_word_tensor = torch.zeros(768)
            else:
                total_word_tensor = feature_extract(each_tweet)[0]
                total_word_tensor = torch.mean(torch.tensor(total_word_tensor), dim=0)
                
            total_each_person_tweets += total_word_tensor

        total_each_person_tweets /= num_tweets
        text_features.append(total_each_person_tweets)

    embedding_tensor = torch.stack(text_features)
    
    embedding_save_path = f'{data_dir}/{dataset}/{feature}/text_feature/tensors/tweets1_tensor_{fro}_{to}.pt'
    os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)

    print("empty tweets:", count, "total tweets:", len(text_features))

    torch.save(embedding_tensor, embedding_save_path)
    del feature_extract 
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    sys.exit(0)

def run_parallel_processes(data, cudas):
    num_processes=len(cudas)
    total = len(data)
    chunk_size = total // num_processes
    
    processes = []
    for i in range(num_processes):
        fro = i * chunk_size
        to = (i + 1) * chunk_size if i < num_processes - 1 else total
        if cudas is None:
            cuda_th = str(i)
        else:
            cuda_th = str(cudas[i])
        p = Process(target=deal, args=(data, cuda_th, fro, to, data_dir, dataset, feature))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

def main(cudas):
    print(data_dir, dataset, feature)
    data_path = f'./deal_dataset/{dataset}/u_tweets_split_feature.json'
    embedding_save_dir = f'{data_dir}/{dataset}/{feature}/text_feature/tensors'

    if os.path.exists(embedding_save_dir):
        os.system(f"rm -rf {embedding_save_dir}")

    positive_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/positive_set.pt").tolist()
    negative_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/negative_set.pt").tolist()
    print(len(positive_set), len(negative_set))

    with open(data_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    txt = []
    idx = 0
    for i in data:
        flag = None
        if idx in positive_set:
            flag = 'positive'
        elif idx in negative_set:
            flag = 'negative'
        if flag:
            txt.append(data[i][f'{feature}_{flag}_tweets'])
            if " ".join(data[i][f'{feature}_{flag}_tweets']).strip() == "":
                print(f"empty tweets: {i}")
        else:
            txt.append([])
        idx += 1
    run_parallel_processes(txt, cudas)

def combine():
    text_features = ["tensors"]
    for text_feature in text_features:
        folder_path = f'{data_dir}/{dataset}/{feature}/text_feature/{text_feature}'
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_paths = [os.path.join(folder_path, f) for f in files]
        save_path = f"{data_dir}/{dataset}/tweets/{feature}/{text_feature}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def extract_number(file_path):
            filename = os.path.basename(file_path)
            numbers = [int(s) for s in filename.split('_') if s.isdigit()]
            return numbers[-1]

        file_paths = sorted(file_paths, key=extract_number)
        print(len(file_paths), len(cudas))
        assert len(file_paths) == len(cudas)

        print(save_path)
        print(file_paths)

        sentiment_dict = []
        for fp in file_paths:
            print(fp)
            tmp = torch.load(fp).tolist()
            sentiment_dict.extend(tmp)

        print(len(sentiment_dict))
        sentiment_dict = torch.tensor(sentiment_dict)

        sentiment_dict = torch.tensor(sentiment_dict)
        all_zero_rows = (sentiment_dict == 0).all(dim=1)
        num_all_zero_vectors = all_zero_rows.sum().item()
        print(f"all 0: {num_all_zero_vectors} / {sentiment_dict.shape[0]}")
        print(f"non-all 0: {sentiment_dict.shape[0] - num_all_zero_vectors}")

        torch.save(sentiment_dict, save_path)
        print(f"save {save_path} done")
 
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_dir = './6FineTune/data/ood2'
    # datasets = ['cresci-2015-data', 'cresci-2017-data', 'twibot-20']
    datasets = ['cresci-2015-data']

    features = ['values', 'emotions']
    for dataset in datasets:
        for feature in features:
            if dataset == 'cresci-2015-data' or dataset == 'cresci-2017-data':
                encoding = 'ISO-8859-1'
            else:
                encoding = 'utf-8'

            cudas = [0, 1, 2, 3, 4, 5, 6, 7]
            main(cudas)
            combine()

