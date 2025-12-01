import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from api_call import api_call
import torch
from experts.emotion_extractor import EmotionExtractor
from experts.sentiment_extractor import SentimentExtractor
from experts.topic_extractor import TopicExtractor
from experts.human_value_extractor import HumanValueExtractor

device = "cuda:0"
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

extractors = {
    "sentiments": sentiment_extractor,
    "topics": topic_extractor,
    "values": human_value_extractor,
    "emotions": emotion_extractor
}

def deal(text):
    if len(text) == 0:
        print(text)
        return " "
    return text.replace("[", "").replace("]", "")

def draw_feature(text):
    extractor = extractors[feature]
    label = extractor(text)
    if label2id[feature][label] in splits[dataset]['description'][feature][0]:
        return "positive"
    elif label2id[feature][label] in splits[dataset]['description'][feature][1]:
        return "negative"
    else:
        return "neutral"

def prompter_train(label, raw_feature, feature, target_feature, text_to_modify):    
    return f"""
    This tweet shows {raw_feature} {feature}, and this tweet's user is a {label}. 
    Please change as few words as possible to rewrite it to a big different {feature} (such as ideally the opposite or ideally unrelated) and don't change the sentence structure to keep the {label} feature.
    Your answer should just contain output without any extra content.
    Input: [{text_to_modify}]
    Output:\n
    """

def prompter_target(label, raw_feature, feature, target_feature, text_to_modify):    
    # strategy: analyze train set and modify the text to other labels' feature. reach a better result. but need more analysis.
    
    return f"""
    This tweet shows {raw_feature} {feature}, and this tweet's user is a bot or human. 
    Please change as few words as possible to rewrite it to big different {feature} such as: {" ".join(target_feature)} and don't change the sentence structure to keep the bot or human feature.
    your answer just contain output without any extra content.
    Input: [{text_to_modify}]
    Output:\n
    """

def prompter_opposite(label, raw_feature, feature, target_feature, text_to_modify):    
    return f"""
    This tweet shows {raw_feature} {feature}, and this tweet's user is a bot or human. 
    Please change as few words as possible to rewrite it to a very different {feature} (such as ideally the opposite or ideally unrelated) and don't change the sentence structure to keep bot or human feature.
    Your answer should just contain output without any extra content.
    Input: [{text_to_modify}]
    Output:\n
    """
    
def prompter(label, raw_feature, feature, target_feature, text_to_modify):
    # simple and effective strategy. thougth it may not such explainable.
    
    return f"""
    This tweet shows {raw_feature} {feature}, and this tweet's user is a bot or human. 
    Please change as few words as possible to rewrite it to a different {feature} and don't change the sentence structure to keep bot or human feature.
    Your answer should just contain output without any extra content.
    Input: [{text_to_modify}]
    Output:\n
    """

def modifiy_one_user(texts_to_modify, prompts):
    if texts_to_modify == -1 or prompts == -1:
        return -1, -1, -1
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_des = {}
        message_prompts = [0 for _ in range(len(texts_to_modify))]
        for i in range(len(texts_to_modify)):
            text_to_modify = texts_to_modify[i]
            prompt = prompts[i]
            message = [
                {
                    "role": "system",
                    "content": "You are an expert in social network analysis, \
                    focusing on the detection of social bots. \
                    You are particularly interested in addressing distribution biases in shallow textual features in {feature} between bots and human users.\
                    Your goal is to modify or augment the text to mitigate this bias while preserving the original semantics."
                },
                {
                    "role": "user",
                    "content": prompter_target(prompt[0], prompt[1], prompt[2], prompt[3], text_to_modify), # reach a better result by more train set analysis.
                    # "content": prompter(prompt[0], prompt[1], prompt[2], prompt[3], text_to_modify)   # reach a more genelization strategy
                }
            ]
            message_prompts[i] = message[1]['content']
            future_to_des[executor.submit(api_call, message)]\
                  = (text_to_modify, i)
        results = [0 for _ in range(len(texts_to_modify))]
        new_features = [0 for _ in range(len(texts_to_modify))]
        for future in as_completed(future_to_des):
            orig_description, idx = future_to_des[future]
            try: 
                content = future.result()
                content = json.loads(content)['choices'][0]['message']['content']
                results[idx] = deal(content)
                new_features[idx] = draw_feature(content)
            except Exception as e:
                results[idx] = str(e)
                new_features[idx] = -1
        return results, new_features, message_prompts

def modifiy_one_dataset(data, positive_set, negative_set, feature):
    modified_text = [0 for i in range(len(data))]
    text_to_modifys = []
    flags = []
    users = list(data.keys())
    prompts = []
    
    for i in range(len(data)):
        if i in positive_set:
            target_set = splits[dataset]['description'][feature][1]
            flag = "positive"
        elif i in negative_set:
            target_set = splits[dataset]['description'][feature][0]
            flag = "negative"
        else:
            text_to_modifys.append(-1)
            prompts.append(-1)
            flags.append(-1)
            continue
        label = data[users[i]]["label"]
        if label == 1:
            label = "bot"
        else:
            label = "human"

        text_to_modify = data[users[i]][f"{feature}_{flag}_tweets"]
        text_to_modifys.append(text_to_modify)
        prompt = []
        raw_features = data[users[i]][f"{feature}_{flag}_tweets_label"]
        for i in range(len(raw_features)):
            prompt.append([label, raw_features[i], feature, target_set])
        prompts.append(prompt)
        flags.append(flag)
    
    test = len(text_to_modifys) + 1
    # test = 10
    text_to_modifys = text_to_modifys[:test]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for i in tqdm(range(len(text_to_modifys))):
            text_to_modify = text_to_modifys[i]    
            prompt = prompts[i]
            futures[executor.submit(modifiy_one_user, text_to_modify, prompt)] = i
        for future in as_completed(futures):
            result, new_feature, message_prompt = future.result()
            i = futures[future]
            modified_text[i] = [result, new_feature, message_prompt]

    modified_dict = {}
    users = users[:test]
    modified_text = modified_text[:test]
    flags = flags[:test]
    for user, text, flag in zip(users, modified_text, flags):
        modified_dict[user] = {}
        modified_dict[user][f"new_tweets"] = text[0]
        modified_dict[user][f"sum_feature"] = flag
        if flag == -1:
            modified_dict[user][f"raw_tweets"] = -1
            modified_dict[user][f"raw_feature"] = -1
        else:
            modified_dict[user][f"raw_tweets"] = data[user][f"{feature}_{flag}_tweets"]
            modified_dict[user][f"raw_feature"] = [flag] * len(data[user][f"{feature}_{flag}_tweets"])
            
        modified_dict[user]["label"] = data[user]["label"]
        modified_dict[user][f"new_feature"] = text[1]
        modified_dict[user][f"message_prompt"] = text[2]

    return modified_dict


with open(f"./data/split_idx/config_trans.json", "r") as f:
    splits = json.load(f)

with open(f"./data/basic/id2label.json", "r") as f:
    id2label = json.load(f)

label2id = {}

for f in id2label:
    label2id[f] = {k:v for v,k in id2label[f].items()}
 
datasets = ["cresci-2015-data"]
features = ["sentiments", "topics", "values", "emotions"]

to_modify_dict = {
    "cresci-2015-data":
    {
        "tweets":
        [
            "sentiments",
            "values",
            "emotions",
            "topics"
        ],
    }
}

to_modify = []
for dataset in to_modify_dict:
    for text in to_modify_dict[dataset]:
        for feature in to_modify_dict[dataset][text]:
            to_modify.append(f'{dataset}_{text}_{feature}')
print(to_modify)

for dataset in datasets:
    with open(f"./deal_dataset/{dataset}/u_tweets_split_feature.json", "r") as f:
        data = json.load(f)
    for feature in features:
        if not f"{dataset}_tweets_{feature}" in to_modify:
            continue
        print(splits[dataset]['description'][feature])
        positive_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/positive_set.pt").tolist()
        negative_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/negative_set.pt").tolist()
        
        test_set = torch.load(f"./data/ood2/{dataset}/tweets/{feature}/00/test_idx.pt").tolist()
        
        modified_dict = modifiy_one_dataset(data, positive_set, negative_set, feature)
        os.makedirs(f"./llm_enhance/{dataset}/tweets/{feature}", exist_ok=True)
        with open(f"./llm_enhance/{dataset}/tweets/{feature}/llm_enhance_modify1.json", "w") as f:
            json.dump(modified_dict, f, indent=4)

