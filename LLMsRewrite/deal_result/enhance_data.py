
"""
main code for mitigating strategies: text level and dataset level
"""

import os
import json
import torch
import multiprocessing as mp
import math

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

import random
import shutil

logging.set_verbosity_error()

def deal(data, cuda_th, fro, to, save_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_th)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    each_user_tweets = data[fro:to]
    feature_extract = pipeline(
        'feature-extraction',
        model='roberta-base',
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
        if tweets == -1 or len(" ".join(tweets).strip()) == 0:
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
    
    embedding_save_path = f'{save_dir}/tweets1_tensor_{fro}_{to}.pt'

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
        save_dir = f'{data_dir}/{dataset}/{feature}/tweets/{mod}/tensors/'
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        p = Process(target=deal, args=(data, cuda_th, fro, to, save_dir))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

def combine():
    text_features = ["tensors"]
    tensor_dir = f"{data_dir}/{dataset}/{feature}/tweets/{mod}"
    for text_feature in text_features:
        folder_path = f'{tensor_dir}/{text_feature}'
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_paths = [os.path.join(folder_path, f) for f in files]
        save_path = f"{tensor_dir}/tensors.pt"
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
        print(f"全为 0 的向量数量: {num_all_zero_vectors} / {sentiment_dict.shape[0]}")
        print(f"非 0 的向量数量: {sentiment_dict.shape[0] - num_all_zero_vectors}")

        torch.save(sentiment_dict, save_path)
        print(f"save {save_path} done")

def count_score(pos, neg, neu):
    _pos = math.exp(pos / (pos + neg + neu))
    _neg = math.exp(neg / (pos + neg + neu))
    _neu = math.exp(neu / (pos + neg + neu))
    return abs(_pos - _neg) / _neu

def select_add(select):
    carry = 1
    for i in range(len(select)):
        tmp_i = (select[i] + carry) % 2
        tmp_carry = (select[i] + carry) // 2
        select[i] = tmp_i
        carry = tmp_carry
    return select

def enhance_one_user(text_raw, raw_feature, text_new, new_feature):
    user_dict = {}
    if text_new == -1 or new_feature == -1:
        user_dict['new_tweets'] = -1
        user_dict['new_feature'] = -1
        user_dict['raw_feature'] = -1
        user_dict['raw_tweets'] = -1
        user_dict['sum_feature'] = -1
        return user_dict
        
    user_dict['raw_tweets'] = text_raw
    user_dict['new_tweets'] = text_new
    modified_tweets = []
    modified_features = []
    if mod == "mix_idx":
        score = 1000
        n = 0
        select = [0] * len(text_raw)
        best_select = [0] * len(text_raw)
        while n < 2**(len(text_raw)):
            valid = 1
            features = [0] * len(text_raw)
            select = select_add(select)
            pos = 0
            neg = 0
            neu = 0
            for i in range(len(select)):

                if select[i] == 0:
                    features[i] = raw_feature[i]
                else:
                    features[i] = new_feature[i]
                
            for i in range(len(features)):
                if features[i] == 'positive':
                    pos += 1
                elif features[i] == 'negative':
                    neg += 1
                elif features[i] == 'neutral':
                    neu += 1
                else:
                    valid = 0
            if valid == 0:
                continue
            score_tmp = count_score(pos, neg, neu)
            
            if score_tmp < score:
                score = score_tmp
                for i in range(len(select)):
                    best_select[i] = select[i]
            n += 1
        
        for i in range(len(best_select)):
            if best_select[i] == 0:
                modified_tweets.append(text_raw[i])
                modified_features.append(raw_feature[i])
            else:
                modified_tweets.append(text_new[i])
                modified_features.append(new_feature[i])
        if sum(best_select) < len(text_raw) // 2:
            modified_tweets = []
        sum_feature = "neutral"
            
    elif mod == "flip_idx":
        for i in range(len(text_raw)):
            if raw_feature[i] == new_feature[i] or new_feature[i] == -1:
                continue
            else:
                modified_tweets.append(text_new[i])
                modified_features.append(new_feature[i])
                if raw_feature[0] == 'positive':
                    sum_feature = "negative"
                elif raw_feature[0] == 'negative':
                    sum_feature = "positive"

    if len(modified_tweets) == 0:
        user_dict['new_tweets'] = -1
        user_dict['new_feature'] = -1
        user_dict['raw_feature'] = -1
        user_dict['raw_tweets'] = -1
        user_dict['sum_feature'] = -1
    else:
        user_dict['new_tweets'] = modified_tweets
        user_dict['new_feature'] = modified_features
        user_dict['raw_feature'] = raw_feature
        user_dict['raw_tweets'] = text_raw
        user_dict['sum_feature'] = sum_feature
    
    return user_dict

def select(data, set_idx):
    modified_data = {}
    with open(f'{data_dir}/{dataset}/{feature}/tweets/flip_idx/modified_tweets.json', 'r') as f:
        flip_data = json.load(f)

    with open(f'{data_dir}/{dataset}/{feature}/tweets/mix_idx/modified_tweets.json', 'r') as f:
        mix_data = json.load(f)
    
    human_pos = []
    human_neg = []
    human_neutral = []
    bot_pos = []
    bot_neg = []
    bot_neutral = []
    idx = 0
    idxs = []
    for u_id in data:
        label = data[u_id]['label']
        if (data[u_id]["raw_tweets"] == -1) or (not idx in set_idx):
            idx += 1
            continue
        else:
            raw_feaure = data[u_id]['raw_feature'][0]
            if flip_data[u_id]["raw_tweets"] == -1:
                tmp = "raw"
                if label == 1 or label == 'bot':
                    if raw_feaure == 'positive':
                        bot_pos.append((idx, "raw"))
                    elif raw_feaure == 'negative':
                        bot_neg.append((idx, "raw"))
                    
                else:
                    if raw_feaure == 'positive':
                        human_pos.append((idx, "raw"))
                    elif raw_feaure == 'negative':
                        human_neg.append((idx, "raw"))
                    
            else:
                tmp = "flip"
                if label == 1 or label == 'bot':
                    if raw_feaure == 'positive':
                        bot_neg.append((idx, "flip"))
                    elif raw_feaure == 'negative':
                        bot_pos.append((idx, "flip"))
                    
                else:
                    if raw_feaure == 'positive':
                        human_neg.append((idx, "flip"))
                    elif raw_feaure == 'negative':
                        human_pos.append((idx, "flip"))
                    
            idxs.append((idx, tmp))
            idx += 1

    scale = 1
    idx_mix = []
    idx_raw = []
    if len(bot_pos) > len(bot_neg)*scale:
        idx_raw.extend(random.sample(bot_pos, len(bot_pos)-(len(bot_pos)+len(bot_neg))//2))
    elif len(bot_neg) > len(bot_pos)*scale:
        idx_mix.extend(random.sample(bot_neg, len(bot_neg)-(len(bot_neg)+len(bot_pos))//2))
    
    if len(human_neg) > len(human_pos)*scale:
        idx_raw.extend(random.sample(human_neg, len(human_neg)-(len(human_neg)+len(human_pos))//2))
    elif len(human_pos) > len(human_neg)*scale:
        idx_mix.extend(random.sample(human_pos, len(human_pos)-(len(human_pos)+len(human_neg))//2))

    for i in range(len(idxs)):
        if idxs[i] in idx_mix:
            idx = idxs[i][0]
            idxs[i] = (idx, "mix")
        elif idxs[i] in idx_raw:
            idx = idxs[i][0]
            idxs[i] = (idx, "raw")


    u_values = list(data.values())
    users = list(data.keys())
    raw_count = 0
    flip_count = 0
    mix_count = 0
    for i in range(len(idxs)):
        idx = idxs[i][0]
        if idxs[i][1] == "raw":
            u_values[idx]['new_tweets'] = data[users[idx]]['raw_tweets']
            u_values[idx]['new_feature'] = data[users[idx]]['raw_feature']
            u_values[idx]['sum_feature'] = data[users[idx]]['sum_feature']
            raw_count += 1
        elif idxs[i][1] == "flip":
            u_values[idx]['new_tweets'] = flip_data[users[idx]]['new_tweets']
            u_values[idx]['new_feature'] = flip_data[users[idx]]['new_feature']
            u_values[idx]['sum_feature'] = flip_data[users[idx]]['sum_feature']
            flip_count += 1
        elif idxs[i][1] == "mix":
            u_values[idx]['new_tweets'] = mix_data[users[idx]]['new_tweets']
            u_values[idx]['new_feature'] = mix_data[users[idx]]['new_feature']
            u_values[idx]['sum_feature'] = mix_data[users[idx]]['sum_feature']
            mix_count += 1
        u_values[idx]['u_id'] = users[idx]
        u_values[idx]['label'] = data[users[idx]]['label']
    
    print(f"{dataset}, {feature}, {mod}, raw_count: {raw_count}, flip_count: {flip_count}, mix_count: {mix_count}")

    for i in range(len(u_values)):
        modified_data[users[i]] = u_values[i]
        
        # move post-processing here
        
        if i not in set_idx:
            modified_data[users[i]]['new_tweets'] = modified_data[users[i]]['raw_tweets']
            modified_data[users[i]]['new_feature'] = modified_data[users[i]]['raw_feature']
            
    with open(f'{data_dir}/{dataset}/{feature}/tweets/{mod}/record.txt', 'w') as f:
        f.write(f"{dataset}, {feature}, {mod}, {len(modified_data)}")
        f.write(f"{dataset}, {feature}, {mod}, raw_count: {raw_count}, flip_count: {flip_count}, mix_count: {mix_count}")
    return modified_data

def enhance_one_dataset(data):
    if os.path.exists(f'{data_dir}/{dataset}/{feature}/tweets/{mod}'):
        shutil.rmtree(f'{data_dir}/{dataset}/{feature}/tweets/{mod}')
    os.makedirs(f'{data_dir}/{dataset}/{feature}/tweets/{mod}', exist_ok=True)
    modified_data = {}
    if mod == "mix_idx" or mod == "flip_idx":
        idx = 0
        for u_id in data:
            # move post-processing here
            if idx not in train_idx:
                modified_data[u_id] = {}
                modified_data[u_id]['u_id'] = u_id
                modified_data[u_id]['label'] = data[u_id]['label']
                modified_data[u_id]['raw_tweets'] = data[u_id]['raw_tweets']
                modified_data[u_id]['raw_feature'] = data[u_id]['raw_feature']
                modified_data[u_id]['new_tweets'] = data[u_id]['raw_tweets']
                modified_data[u_id]['new_feature'] = data[u_id]['raw_feature']
                modified_data[u_id]['sum_feature'] = data[u_id]['sum_feature']
                idx += 1
                continue
            text_raw = data[u_id]['raw_tweets']
            text_new = data[u_id]['new_tweets']
            raw_feature = data[u_id]['raw_feature']
            new_feature = data[u_id]['new_feature']
            modified_data[u_id] = {}
            modified_data[u_id] = enhance_one_user(text_raw, raw_feature, text_new, new_feature)
            modified_data[u_id]['u_id'] = u_id
            modified_data[u_id]['label'] = data[u_id]['label']
            idx += 1

    if mod == "select_idx":
        modified_data = select(data, train_idx)

    with open(f'{data_dir}/{dataset}/{feature}/tweets/{mod}/modified_tweets.json', 'w') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=4)
    
    return modified_data

def main(cudas):
    with open(f"{data_dir}/{dataset}/tweets/{feature}/llm_enhance_modify1.json", "r") as f:
        data = json.load(f)
    
    modified_data = enhance_one_dataset(data)

    texts = []
    for u_id in modified_data:
        texts.append(modified_data[u_id]['new_tweets'])
        
    run_parallel_processes(texts, cudas)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_dir = './llm_enhance'
    datasets = ['cresci-2015-data']
    # datasets = ['cresci-2015-data', 'cresci-2017-data', 'twibot-20']

    mods = ["flip_idx", "mix_idx", "select_idx"]
    features = ["sentiments", "topics", "emotions", "values"]

    
    for dataset in datasets:
        for feature in features:
            for mod in mods:
            
                train_idx = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/00/train_idx.pt").tolist()
                test_idx = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/00/test_idx.pt").tolist()

                cudas = [0, 1, 2, 3, 4, 5, 6, 7]
                main(cudas)
                combine()
                