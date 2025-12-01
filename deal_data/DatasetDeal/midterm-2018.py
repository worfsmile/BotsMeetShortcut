import json
import csv
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

with open('deal_dataset/midterm-2018/midterm-2018_processed_user_objects.json', 'r') as f:
    user_objects = json.load(f)

label_dict = {}
with open('deal_dataset/midterm-2018/midterm-2018.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if row[1] == 'bot':
            label_dict[row[0]] = 1
        else:
            label_dict[row[0]] = 0

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

u_des_tweets = {}
print(len(user_objects))
print(len(label_dict))
for i in user_objects:
    u_id = i['user_id']
    u_des_tweets[u_id] = {}
    u_des_tweets[u_id]['u_id'] = str(u_id)
    u_des_tweets[u_id]['label'] = label_dict[str(u_id)]
    u_des_tweets[u_id]['followers_count'] = safe_int(i['followers_count'])
    u_des_tweets[u_id]['statuses_count'] = safe_int(i['statuses_count'])
    u_des_tweets[u_id]['friends_count'] = safe_int(i['friends_count'])
    u_des_tweets[u_id]['listed_count'] = safe_int(i['listed_count'])
    u_des_tweets[u_id]['favourites_count'] = safe_int(i['favourites_count'])
    u_des_tweets[u_id]['screen_name_length'] = len(i['screen_name'])
    u_des_tweets[u_id]['name_length'] = len(i['name'])
    u_des_tweets[u_id]['screen_name_sim'] = LCSubstr(i['screen_name'], i['name'])
    u_des_tweets[u_id]['description'] = convert_non_ascii_to_unicode(i['description']) if i['description'] else ''
    u_des_tweets[u_id]['tweets'] = []

print(len(u_des_tweets))
with open('./deal_dataset/midterm-2018/u_tweets.json', 'w') as f:
    json.dump(u_des_tweets, f, indent=4)
    
