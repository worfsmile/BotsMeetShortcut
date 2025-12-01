# convert the user comments into AMRs
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from multiprocessing import Process, cpu_count
import glob
import amrlib
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import traceback
import torch
import multiprocessing as mp

def check_file_in_folder(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return True
    except:
        return False

def deal(data, args, cuda_th = "0", fro = 0, to = 0):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_th
    stog = amrlib.load_stog_model("your text2graph model path")
    print("Loaded stog")
    if args.text == "tweets":
        positive_set = torch.load(f"./data/ood2/{args.dataset}/tweets/{args.feature}/positive_set.pt").tolist()
        negative_set = torch.load(f"./data/ood2/{args.dataset}/{args.text}/{args.feature}/negative_set.pt").tolist()
    check = 1
    count = 0
    for lv in range(fro, to):
        u_dict, idx = data[lv]
        dest_dir = f'./8AMR_CIGA/data/amr_gen/{args.dataset}/{args.text}/{args.feature}/'
        os.makedirs(dest_dir, exist_ok = True)
        save_path = dest_dir+str(u_dict['u_id'])+'.json'
        if check_file_in_folder(dest_dir, str(u_dict['u_id'])+'.json'):
            if check:
                check = 0
                print("checked")
            continue
        if args.text == "tweets":
            if idx in positive_set:
                raw_tweets = u_dict[f'{args.feature}_positive_tweets']
            elif idx in negative_set:
                raw_tweets = u_dict[f'{args.feature}_negative_tweets']
            else:
                raise ValueError("Invalid index")
        else:
            raw_tweets = u_dict[args.text]

        if not isinstance(raw_tweets, list):
            raw_tweets = [raw_tweets]

        tweets = raw_tweets
        try:
            graphs = stog.parse_sents(tweets)
            count += 1
        except Exception as e:
            error_message = traceback.format_exc()
            with open(f"error_log{args.dataset}.txt", "a") as log_file:
                log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {args.dataset}{str(u_dict['u_id'])}, Error processing tweets: {error_message}\n")
            continue
        
        u_amr = {"u_id":str(u_dict['u_id']), "raw_tweets":raw_tweets, "amr":graphs}
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(u_amr, json_file, ensure_ascii=False, indent=4)
        print(f"Done {lv}/{fro}-{to}")
    print(f"Done{fro}-{to}:", count, fro-to)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
        
def run_parallel_processes(data, args, cudas = None):
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
        p = Process(target=deal, args=(data, args, cuda_th, fro, to))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

def main(args, raw_data_root, cudas):
    path = os.path.join(raw_data_root, args.dataset, f'u_tweets_split_feature.json')
    if not os.path.exists(path):
        print(f"Raw data not found at {path}")
        return
    dest_dir = f'./8AMR_CIGA/data/amr_gen/{args.dataset}/{args.dataset}_amr/'
    if not os.path.exists(f"./data/ood2/{args.dataset}/{args.text}/{args.feature}/00/train_idx.pt"):
        print(f"Train idx not found")
        return
    train_idx00 = torch.load(f"./data/ood2/{args.dataset}/{args.text}/{args.feature}/00/train_idx.pt").tolist()
    test_idx00= torch.load(f"./data/ood2/{args.dataset}/{args.text}/{args.feature}/00/test_idx.pt").tolist()
    train_idx11 = torch.load(f"./data/ood2/{args.dataset}/{args.text}/{args.feature}/11/train_idx.pt").tolist()
    test_idx11= torch.load(f"./data/ood2/{args.dataset}/{args.text}/{args.feature}/11/test_idx.pt").tolist()
    train_idx = train_idx00 + train_idx11
    test_idx = test_idx00 + test_idx11
    idxs = train_idx + test_idx
    with open(path, encoding='utf-8') as f:
        all_data = json.load(f)
    all_data = list(all_data.values())
    data = []
    for lv in range(len(all_data)):
        if check_file_in_folder(dest_dir, str(all_data[lv]['u_id'])+'.json') or lv not in idxs:
            continue
        data.append([all_data[lv],lv])
    print(len(data))
    run_parallel_processes(data, args, cudas)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    datasets = ["cresci-stock-2018", "midterm-2018", "twibot-20"]
    texts = ['description', 'tweets']
    features = ['sentiments', 'emotions', 'topics', 'values']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = '', choices = [], help='Specify the dataset for which you want to run the experiments.')
    parser.add_argument('--feature', default = 'sentiments', choices = ['sentiments', 'emotions', 'topics', 'values'], help='Specify the feature to use for generating AMRs.')
    parser.add_argument('--text', default = 'description', choices = ['description', 'tweets'], help='Specify the text field to use for generating AMRs.')
    parser.add_argument('--batch-size', type=int, default=10, help="Number of posts to process in one batch.")
    args = parser.parse_args()
    raw_data_root = './deal_dataset'
    for data_name in datasets:
        for text in texts:
            if text == 'description':
                num = 5
            else:
                num = 1
            for feature in features:
                args.text = text
                args.feature = feature
                args.dataset = data_name
                print(data_name)

                main(args, raw_data_root, cudas=[0,1,2,3,4,5,6,7]*num)
                print("done", data_name)
        
