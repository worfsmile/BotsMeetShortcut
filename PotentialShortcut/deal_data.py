import os
import json
import torch
from experts.emotion_extractor import EmotionExtractor
from experts.sentiment_extractor import SentimentExtractor
from experts.topic_extractor import TopicExtractor
from experts.human_value_extractor import HumanValueExtractor
from experts.feature_extractor import FeatureExtractor

import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import shutil

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

with open("data/split_idx/config_trans.json", 'r') as f:
    spilt_idx = json.load(f)


def deal(data, dataset, cuda_th, parallel_idx):
    print(f"deal {dataset} {parallel_idx}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_th)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiments_positive, sentiments_negative, sentiments_neutral = spilt_idx[dataset]['description']['sentiments']
    topics_positive, topics_negative, topics_neutral = spilt_idx[dataset]['description']['topics']
    emotions_positive, emotions_negative, emotions_neutral = spilt_idx[dataset]['description']['emotions']
    values_positive, values_negative, values_neutral = spilt_idx[dataset]['description']['values']
    
    sentiment_extractor = SentimentExtractor(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device)
    emotion_extractor = EmotionExtractor(
        "cardiffnlp/twitter-roberta-large-emotion-latest",
        device=device)
    topic_extractor = TopicExtractor(
        "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all",
        device=device)
    human_value_extractor = HumanValueExtractor(
        "victorYeste/deberta-based-human-value-detection",
        device=device)
    
    def deal_one_sentence(sentence):
        sentiments = sentiment_extractor(sentence)
        topics = topic_extractor(sentence)
        emotions = emotion_extractor(sentence)
        values = human_value_extractor(sentence)
        sentiments = sentiment_extractor.label_dict[sentiments]
        topics = topic_extractor.label_dict[topics]
        emotions = emotion_extractor.label_dict[emotions]
        values = human_value_extractor.label_dict[values]
        return sentiments, topics, emotions, values

    def deal_one_user(user_dict):
        u_dict = {}
        discription = user_dict['description']
        u_dict['label'] = user_dict['label']
        u_dict['u_id'] = user_dict['u_id']
        u_dict['description'] = discription
        discription_sentiments, discription_topics, discription_emotions, discription_values = deal_one_sentence(discription)
        u_dict['description_sentiments'] = discription_sentiments
        u_dict['description_topics'] = discription_topics
        u_dict['description_emotions'] = discription_emotions
        u_dict['description_values'] = discription_values
        u_dict['sentiments_positive_tweets'] = []
        u_dict['sentiments_positive_tweets_label'] = []
        u_dict['sentiments_negative_tweets'] = []
        u_dict['sentiments_negative_tweets_label'] = []
        u_dict['topics_positive_tweets'] = []
        u_dict['topics_positive_tweets_label'] = []
        u_dict['topics_negative_tweets'] = []
        u_dict['topics_negative_tweets_label'] = []
        u_dict['emotions_positive_tweets'] = []
        u_dict['emotions_positive_tweets_label'] = []
        u_dict['emotions_negative_tweets'] = []
        u_dict['emotions_negative_tweets_label'] = []
        u_dict['values_positive_tweets'] = []
        u_dict['values_positive_tweets_label'] = []
        u_dict['values_negative_tweets'] = []
        u_dict['values_negative_tweets_label'] = []
        tweets = user_dict['tweets']
        tweets = random.sample(tweets, min(len(tweets), 100))
        for i in range(len(tweets)):
            tweet = tweets[i]
            sentiments, topics, emotions, values = deal_one_sentence(tweet)
            if sentiments in sentiments_positive and len(u_dict['sentiments_positive_tweets']) < 5:
                u_dict['sentiments_positive_tweets'].append(tweet)
                u_dict['sentiments_positive_tweets_label'].append(sentiments)
            elif sentiments in sentiments_negative and len(u_dict['sentiments_negative_tweets']) < 5:
                u_dict['sentiments_negative_tweets'].append(tweet)
                u_dict['sentiments_negative_tweets_label'].append(sentiments)
            if topics in topics_positive and len(u_dict['topics_positive_tweets']) < 5:
                u_dict['topics_positive_tweets'].append(tweet)
                u_dict['topics_positive_tweets_label'].append(topics)
            elif topics in topics_negative and len(u_dict['topics_negative_tweets']) < 5:
                u_dict['topics_negative_tweets'].append(tweet)
                u_dict['topics_negative_tweets_label'].append(topics)
            if emotions in emotions_positive and len(u_dict['emotions_positive_tweets']) < 5:
                u_dict['emotions_positive_tweets'].append(tweet)
                u_dict['emotions_positive_tweets_label'].append(emotions)
            elif emotions in emotions_negative and len(u_dict['emotions_negative_tweets']) < 5:
                u_dict['emotions_negative_tweets'].append(tweet)
                u_dict['emotions_negative_tweets_label'].append(emotions)
            if values in values_positive and len(u_dict['values_positive_tweets']) < 5:
                u_dict['values_positive_tweets'].append(tweet)
                u_dict['values_positive_tweets_label'].append(values)
            elif values in values_negative and len(u_dict['values_negative_tweets']) < 5:
                u_dict['values_negative_tweets'].append(tweet)
                u_dict['values_negative_tweets_label'].append(values)
            if len(u_dict['sentiments_positive_tweets']) >= 5 and len(u_dict['sentiments_negative_tweets']) >= 5\
                and len(u_dict['topics_positive_tweets']) >= 5 and len(u_dict['topics_negative_tweets']) >= 5\
                and len(u_dict['emotions_positive_tweets']) >= 5 and len(u_dict['emotions_negative_tweets']) >= 5\
                and len(u_dict['values_positive_tweets']) >= 5 and len(u_dict['values_negative_tweets']) >= 5:
                break
        return u_dict
    
    deal_data = {}
    
    for user in tqdm(data, desc="Processing users"):
        u_dict = deal_one_user(user)
        deal_data[user['u_id']] = u_dict
        
    tmp_dir = f"deal_dataset/{dataset}/{dataset}/tmp"

    os.makedirs(tmp_dir, exist_ok=True)
    with open(f"deal_dataset/{dataset}/{dataset}/tmp/deal_data_{parallel_idx}.json", 'w') as f:
        json.dump(deal_data, f)

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
        p = Process(target=deal, args=(data[fro:to], dataset, cuda_th, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

def main(data_path, cudas):
    with open(data_path, 'r', encoding=encoding) as f:
        datas = json.load(f)
    txt = []
    for i in datas:
        txt.append(datas[i])
    run_parallel_processes(txt, cudas)

def combine():
    folder_path = f'{data_dir}/{dataset}/{dataset}/tmp'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    file_paths = [os.path.join(folder_path, f) for f in files]
    save_path = f"{data_dir}/{dataset}/u_tweets_split_feature.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def extract_number(file_path):
        filename = os.path.basename(file_path).split('.')[0]
        numbers = [int(s) for s in filename.split('_') if s.isdigit()]
        return numbers[-1]

    file_paths = sorted(file_paths, key=extract_number)
    print(len(file_paths), len(cudas))
    assert len(file_paths) == len(cudas)

    print(save_path)
    print(file_paths)

    sentiment_dict = {}
    for fp in file_paths:
        print(fp)
        with open(fp, 'r') as f:
            tmp = json.load(f)
        sentiment_dict.update(tmp)

    print(len(sentiment_dict))

    with open(save_path, 'w') as f:
        json.dump(sentiment_dict, f, indent=4)

if __name__ == "__main__":
    data_dir = './deal_dataset'
    # datasets = ['cresci-2015-data','cresci-2017-data', 'cresci-stock-2018', 'midterm-2018', 'twibot-20', 'twibot-22']

    datasets = ['cresci-2017-data']
    cudas = [1, 2, 3, 4, 5, 6, 7] * 3
    # cudas = [0] * 3
    for dataset in datasets:
        if dataset == 'cresci-2015-data' or dataset == 'cresci-2017-data':
            encoding = 'ISO-8859-1'
        else:
            encoding = 'utf-8'
        
        data_path = os.path.join(data_dir, dataset, 'u_tweets.json')
        main(data_path, cudas)
        combine()
