import ijson
import pandas as pd
import re
import csv
import os
import json
import sys

def convert_non_ascii_to_unicode(text):
    text = text.strip()
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\n\t', ' ', text)
    text = re.sub(r'\t\n', ' ', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'(\s|^)\(', r' (', text)
    return text

def safe_int(value, default=0):
    try:
        if isinstance(value, str):
            if "False" in value:
                value = 0
            elif "True" in value:
                value = 1
        return int(value)
    except (ValueError, TypeError):
        return default

def LCSubstr(s1, s2):
    m, n = len(s1), len(s2)
    # Create a DP table with (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0  # Variable to store the length of the longest common substring
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])  # Update the max length
            else:
                dp[i][j] = 0  # If characters don't match, reset to 0
    return max_length / min(m, n) if m > 0 and n > 0 else 0

def translabel(label):
    if label == 'bot' or label == '1':
        return 1
    if label == 'human'or label =='0':
        return 0
    return label

def node2json(file_path, output_file='output.json'):
    id_label_dict = pd.read_csv("path_to/twibot20/label.csv")
    print("read label")
    edges_post_dict = pd.read_csv("path_to/twibot20/edge.csv")
    print("read edge")
    edges_post_dict = edges_post_dict[edges_post_dict['relation'] == 'post']
    edges_post_dict = dict(zip(edges_post_dict['target_id'], edges_post_dict['source_id']))
    id_label_dict = dict(zip(id_label_dict['id'], id_label_dict['label']))
    print("begin")
    nowuser = None
    u_tweets = {}
    u_description = {}
    count = 0
    towrite = 0
    flag = 1
    item_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        print("open")
        for item in ijson.items(f, 'item'):
            item_count += 1
            if item_count % 10000 == 0:
                print(item_count)
            if count % 10000 == 0 and flag:
                print(count)
                flag = 0
            try:
                if item['id'].startswith('u'):
                    public_metrics = item['public_metrics']
                    
                    u_tweets[item['id']] = {
                    "u_id": item['id'],
                    "label": translabel(id_label_dict.get(item['id'], -1)),
                    "tweets": [],
                    "description": convert_non_ascii_to_unicode(item['description']) if item['description'] else '',
                    "followers_count": safe_int(public_metrics.get('followers_count', None)),
                    "friends_count": safe_int(public_metrics.get('following_count', None)),
                    "listed_count": safe_int(public_metrics.get('listed_count', None)),
                    "statuses_count": safe_int(public_metrics.get('tweet_count', None)),
                    "screen_name_length": len(item['username']),
                    "name_length": len(item['name']),
                    "screen_name_sim": LCSubstr(item['username'], item['name']),
                    "verified": safe_int(item.get('verified', -1), default=-1),
                    "protected": safe_int(item.get('protected', -1), default=-1),
                }
                
                elif item['id'] in edges_post_dict:
                    if edges_post_dict[item['id']] in u_tweets:
                        nowuser = edges_post_dict[item['id']]
                        if len(u_tweets[nowuser]['tweets']) < 100:
                            u_tweets[nowuser]['tweets'].append(convert_non_ascii_to_unicode(item['text']))
            except Exception as e:
                print(e)
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='') as file:
        json.dump(u_tweets, file, ensure_ascii=False, indent=4)
    print(len(u_tweets))
    print(count)


path = "./deal_dataset/twibot20/node.json"

node2json(path, './deal_dataset/twibot-20/u_tweets.json')
