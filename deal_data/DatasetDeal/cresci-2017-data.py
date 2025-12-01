import os
import pandas as pd
import re
import json

count = [0]

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

def load_data(path, user_path, label, extract_label):
    tweets_df = pd.read_csv(path, encoding='ISO-8859-1', dtype=str).fillna('')
    user_df = pd.read_csv(user_path, encoding='ISO-8859-1', dtype=str).fillna('')
    
    id_description_dict = dict(zip(user_df['id'], user_df['description']))
    id_fans = dict(zip(user_df['id'], user_df['followers_count']))
    id_statuses = dict(zip(user_df['id'], user_df['statuses_count']))
    id_friends = dict(zip(user_df['id'], user_df['friends_count']))
    id_listed = dict(zip(user_df['id'], user_df['listed_count']))
    id_favorites = dict(zip(user_df['id'], user_df['favourites_count']))
    id_screen_name = dict(zip(list(map(str, list(map(int, user_df['id'])))), user_df['screen_name']))
    id_name = dict(zip(list(map(str, list(map(int, user_df['id'])))), user_df['name']))
    
    u_tweets = {}
    for i, row in tweets_df.iterrows():
        u_id = row["user_id"]
        if pd.isna(u_id):  # Check if u_id is NaN
            continue
        count[0] += 1
        if u_id not in u_tweets:
            if u_id == '':
                continue
            description = convert_non_ascii_to_unicode(id_description_dict[u_id])
            description = description.strip()
            str_u_id = str(int(float(u_id)))
            u_tweets[u_id] = {"u_id": str_u_id,
                              "label": label,
                              "followers_count": safe_int(id_fans[u_id]),
                              "statuses_count": safe_int(id_statuses[u_id]),
                              "friends_count": safe_int(id_friends[u_id]),
                              "listed_count": safe_int(id_listed[u_id]),
                              "favourites_count": safe_int(id_favorites[u_id]),
                              "screen_name_length": len(id_screen_name[str_u_id]),
                              "name_length": len(id_name[str_u_id]),
                              "screen_name_sim": LCSubstr(id_screen_name[str_u_id], id_name[str_u_id]),
                              "description": description,
                              "tweets": [],
                              }
            
            if not isinstance(u_tweets[u_id]["description"], str):
                u_tweets[u_id]["description"] = ""
                
        text = row["text"]
        if not isinstance(text, str):
            text = ""
            
        text = convert_non_ascii_to_unicode(text)
        text = text.strip()    
        u_tweets[u_id]["tweets"].append(text)
    return u_tweets

def load_per_data(path):
    hunam_file = ["genuine_accounts"]
    bot_file = ["fake_followers", "social_spambots_1", "social_spambots_2", "social_spambots_3", "traditional_spambots_1"]
    u_tweets = {}
    for file in hunam_file:
        u_tweets.update(load_data(os.path.join(path, file+".csv", file+".csv", "tweets.csv"),
                                  os.path.join(path, file+".csv", file+".csv", "users.csv"), 0, file))
        print("done", file)
    for file in bot_file:
        u_tweets.update(load_data(os.path.join(path, file+".csv", file+".csv", "tweets.csv"),
                                  os.path.join(path, file+".csv", file+".csv", "users.csv"), 1, file))
        print("done", file)

    save_data_to_json(u_tweets, path)

def save_data_to_json(data, dir_path):
    if len(data) == 0:
        print("No data to save")
        return
    if not os.path.exists(os.path.dirname(dir_path)):
        os.makedirs(os.path.dirname(dir_path))
    
    save_path = os.path.join(dir_path, "u_tweets.json")
    with open(save_path, 'w', encoding='ISO-8859-1') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)  # Save the data to JSON, ensure non-ASCII chars are preserved

path = "./deal_dataset/cresci-2017-data"

load_per_data(path)
print("Total tweets:", count[0])

