import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentExtractor:
    def __init__(self, senti_model_path, device):
        self.senti_tokenizer = AutoTokenizer.from_pretrained(senti_model_path)
        self.senti_model = AutoModelForSequenceClassification.from_pretrained(senti_model_path).to(device)
        self.device = device
        self.label_dict = {-1: "null", 0: "negative", 1: "neutral", 2: "positive"}

    def __call__(self, text):
        if text == "" or text is None:
            return -1
        else:
            inputs = self.senti_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.senti_model(**inputs)
            logits = outputs.logits
            return logits.argmax().item()

