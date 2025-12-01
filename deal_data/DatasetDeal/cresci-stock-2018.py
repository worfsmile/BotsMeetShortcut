import csv
import json
import pandas as pd
import re

def convert_non_ascii_to_unicode(text):
    text = text.strip()
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\n\t', ' ', text)
    text = re.sub(r'\t\n', ' ', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'(\s|^)\(', r' (', text)
    return text

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

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

label_dict = {}
with open('deal_dataset/cresci-stock-2018/cresci-stock-2018.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if row[1] == 'bot':
            label_dict[str(row[0])] = 1
        else:
            label_dict[str(row[0])] = 0

u_tweets_dict = {}

with open('deal_dataset/cresci-stock-2018/cresci-stock-2018_tweets.json', 'r', encoding='utf-8') as f:
    tweets_data = json.load(f)

# df = pd.read_csv('your_file.csv', dtype=str)

user_df = pd.read_csv('deal_dataset/cresci-stock-2018/cresci-stock-2018_extracted_data.csv', encoding='ISO-8859-1', dtype=str).fillna('')

id_fans = dict(zip(user_df['id'], user_df['followers_count']))
id_statuses = dict(zip(user_df['id'], user_df['statuses_count']))
id_friends = dict(zip(user_df['id'], user_df['friends_count']))
id_listed = dict(zip(user_df['id'], user_df['listed_count']))
id_favorites = dict(zip(user_df['id'], user_df['favourites_count']))

u_description_dict = {}
sreen_name_dict = {}
name_dict = {}

for i in range(len(tweets_data)):
    if tweets_data[i]['user']['id'] not in u_description_dict:
        u_description_dict[str(int(tweets_data[i]['user']['id']))] = convert_non_ascii_to_unicode(tweets_data[i]['user']['description'])
        sreen_name_dict[str(int(tweets_data[i]['user']['id']))] = convert_non_ascii_to_unicode(tweets_data[i]['user']['screen_name'])
        name_dict[str(int(tweets_data[i]['user']['id']))] = convert_non_ascii_to_unicode(tweets_data[i]['user']['name'])
    else:
        print('error')
    

for i, row in user_df.iterrows():
    u_id = str(int(row['id']))
    description = u_description_dict[u_id]
    u_tweets_dict[u_id] = {
        "u_id": u_id,
        "label": label_dict[str(int(u_id))],
        "followers_count": safe_int(id_fans[u_id]),
        "statuses_count": safe_int(id_statuses[u_id]),
        "friends_count": safe_int(id_friends[u_id]),
        "listed_count": safe_int(id_listed[u_id]),
        "favourites_count": safe_int(id_favorites[u_id]),
        "screen_name_length": len(sreen_name_dict[u_id]),
        "name_length": len(name_dict[u_id]),
        "screen_name_sim": LCSubstr(sreen_name_dict[u_id], name_dict[u_id]),
        "description": description,
        "tweets": [],
    }

with open('./deal_dataset/cresci-stock-2018/u_tweets.json', 'w', encoding='utf-8') as f:
    json.dump(u_tweets_dict, f, ensure_ascii=False, indent=2)
