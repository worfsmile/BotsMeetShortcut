import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class EmotionExtractor:
    def __init__(self, ckpt, device):
        self.ckpt = ckpt
        self.device = device
        self.emotion_dict = {
            'null': -1,
            'anger': 0,
            'anticipation': 1,
            'disgust': 2,
            'fear': 3,
            'joy': 4,
            'love': 5,
            'optimism': 6,
            'pessimism': 7,
            'sadness': 8,
            'surprise': 9,
            'trust': 10
        }
        self.pipe = pipeline('text-classification', model=self.ckpt, tokenizer=self.ckpt, padding=True, truncation=True, max_length=512, return_all_scores=True, device=device)
        self.label_dict = {v: k for k, v in self.emotion_dict.items()}

    def __call__(self, text):   #string -> int
        if text == '' or text is None:
            predictions = -1
        else:
            predictions = self.pipe(text)[0]
            predictions = self.emotion_dict[max(predictions, key=lambda x: x['score'])['label']]
        return predictions

