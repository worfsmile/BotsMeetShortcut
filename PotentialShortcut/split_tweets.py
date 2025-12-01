import random
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
import os
import pandas as pd

limit = 5000

def classify_one_user(user, feature):
    # print(user)
    flag = ""
    for i in ['positive', 'negative']:
        if len(user[f"{feature}_{i}_tweets"]) == 0:
            flag += "0"
        else:
            flag += "1"
    return flag

def count(data, feature):
    positive_human = []
    negative_human = []
    both_human = []
    none_human = []
    positive_bot = []
    negative_bot = []
    both_bot = []
    none_bot = []
    idx = 0
    for user_id in data:
        user = data[user_id]
        flag = classify_one_user(user, feature)
        if flag == "11":
            if user['label'] == 0:
                both_human.append(idx)
            else:
                both_bot.append(idx)
        elif flag == "00":
            if user['label'] == 0:
                none_human.append(idx)
            else:
                none_bot.append(idx)
        elif flag == "10":
            if user['label'] == 0:
                positive_human.append(idx)
            else:
                positive_bot.append(idx)
        elif flag == "01":
            if user['label'] == 0:
                negative_human.append(idx)
            else:
                negative_bot.append(idx)
        idx += 1
    return positive_human, negative_human, both_human, none_human, positive_bot, negative_bot, both_bot, none_bot

def balance_data(positive_human, negative_human, both_human):
    human_positive = positive_human
    human_negative = negative_human
    diff = len(positive_human) - len(negative_human)
    mid = len(positive_human) // 2
    if diff > 0:
        extend_set = random.sample(both_human, min(diff, len(both_human)))
        human_negative.extend(extend_set)
    else:
        extend_set = random.sample(both_human, min(-diff, len(both_human)))
        human_positive.extend(extend_set)
    
    both_human = list(set(both_human) - set(extend_set))
    random.shuffle(both_human)
    mid2 = len(both_human) // 2
    human_negative.extend(both_human[mid2:])
    human_positive.extend(both_human[:mid2])
    limited = min(len(human_positive), len(human_negative))
    human_positive = random.sample(human_positive, limited)
    human_negative = random.sample(human_negative, limited)
    return human_positive, human_negative

def split_data(data, feature, idx_dir, dataset):
    positive_human, negative_human, both_human, none_human, positive_bot, negative_bot, both_bot, none_bot = count(data, feature)
    print(feature,
          "\npositive_human:", len(positive_human),
          "\nnegative_human:", len(negative_human),
          "\nboth_human:", len(both_human),
          "\nnone_human:", len(none_human),
          "\npositive_bot:", len(positive_bot),
          "\nnegative_bot:", len(negative_bot),
          "\nboth_bot:", len(both_bot),
          "\nnone_bot:", len(none_bot))
    
    all_human_count = len(positive_human) + len(negative_human) + len(both_human) + len(none_human)
    all_bot_count = len(positive_bot) + len(negative_bot) + len(both_bot) + len(none_bot)
    human_positive, human_negative = balance_data(positive_human, negative_human, both_human)
    bot_positive, bot_negative = balance_data(positive_bot, negative_bot, both_bot)
    
    limited1 = min(len(human_positive), len(bot_negative), limit)
    limited2 = min(len(human_negative), len(bot_positive), limit)
    
    human_positive = random.sample(human_positive, limited1)
    human_negative = random.sample(human_negative, limited1)
    bot_positive = random.sample(bot_positive, limited2)
    bot_negative = random.sample(bot_negative, limited2)
    print("human_positive:", len(human_positive),
          "\nhuman_negative:", len(human_negative),
          "\nbot_positive:", len(bot_positive),
          "\nbot_negative:", len(bot_negative))
    
    assert len(set(human_positive) & set(human_negative)) == 0

    positive_set = human_positive + bot_positive
    negative_set = human_negative + bot_negative

    human_positive_train, human_positive_test = train_test_split(human_positive, test_size=0.2, random_state=42)
    human_negative_train, human_negative_test = train_test_split(human_negative, test_size=0.2, random_state=42)
    bot_positive_train, bot_positive_test = train_test_split(bot_positive, test_size=0.2, random_state=42)
    bot_negative_train, bot_negative_test = train_test_split(bot_negative, test_size=0.2, random_state=42)

    train_ood = human_positive_train + bot_negative_train
    test_ood = human_negative_test + bot_positive_test

    limited3 = min(len(human_positive_train), len(human_negative_train), len(bot_positive_train), len(bot_negative_train)) // 2

    train_id = random.sample(human_positive_train, limited3)+ \
               random.sample(human_negative_train, limited3)+ \
               random.sample(bot_positive_train, limited3)+ \
               random.sample(bot_negative_train, limited3)
    
    limited4 = min(len(human_positive_test), len(human_negative_test), len(bot_positive_test), len(bot_negative_test))
    test_id = random.sample(human_positive_test, limited4)+ \
              random.sample(human_negative_test, limited4)+ \
              random.sample(bot_positive_test, limited4)+ \
              random.sample(bot_negative_test, limited4)
    
    train_ood = torch.tensor(train_ood)
    test_ood = torch.tensor(test_ood)
    train_id = torch.tensor(train_id)
    test_id = torch.tensor(test_id)

    idxs = [
        [train_ood, train_id],
        [test_ood, test_id],
    ]

    for i in range(2):
        for j in range(2):
            train_set = idxs[0][i]
            test_set = idxs[1][j]
            # 
            if os.path.exists(f"{idx_dir}/{feature}/{i}{j}"):
                os.system(f"rm -rf {idx_dir}/{feature}/{i}{j}")
            os.makedirs(f"{idx_dir}/{feature}/{i}{j}", exist_ok=True)
            torch.save(train_set, f"{idx_dir}/{feature}/{i}{j}/train_idx.pt")
            torch.save(test_set, f"{idx_dir}/{feature}/{i}{j}/test_idx.pt")

    positive_set = torch.tensor(positive_set)
    negative_set = torch.tensor(negative_set)
    torch.save(positive_set, f"{idx_dir}/{feature}/positive_set.pt")
    torch.save(negative_set, f"{idx_dir}/{feature}/negative_set.pt")

    stat_entry = {
    "dataset": dataset,
    "text": "tweets",
    "feature": feature,
    "configs": {
        '00': (len(train_ood), len(test_ood)),
        '01': (len(train_ood), len(test_id)),
        '10': (len(train_id), len(test_ood)),
        '11': (len(train_id), len(test_id))
        }
    }
    stats.append(stat_entry)
    

if __name__ == '__main__':
    # datasets = ['cresci-2015-data', 'cresci-2017-data', 'twibot-20']
    datasets = ['cresci-2017-data']

    features = ['sentiments', 'emotions', 'topics', 'values']
    stats = []
    for dataset in datasets:
        for feature in features:
            with open(f"./deal_dataset/{dataset}/u_tweets_split_feature.json", "r") as f:
                data = json.load(f)
            idx_dir = f"./data/ood2/{dataset}/tweets"
            split_data(data, feature, idx_dir, dataset)

    table_data = []
    for entry in stats:
        for config in ['00', '01', '10', '11']:
            train_size, test_size = entry["configs"][config]
            table_data.append({
                "Dataset": entry["dataset"],
                "Text": entry["text"],
                "Feature": entry["feature"],
                "Config": config,
                "Train Size": train_size,
                "Test Size": test_size,
                "Total": train_size + test_size
            })

    df = pd.DataFrame(table_data)
    df.to_csv('./data/split_idx/ood2.csv', index=False)



    

