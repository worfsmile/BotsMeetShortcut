from transformers import pipeline
import torch

class FeatureExtractor:
    def __init__(self, model_name_or_path, device=None):
        self.device = device
        self.feature_extract = pipeline(
            'feature-extraction',
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device=device,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )

    def __call__(self, text):   #string -> tensor
        if text == '' or text is None:
            feature = torch.zeros(768)
        else:
            feature = self.feature_extract(text)[0]
            feature = torch.mean(torch.tensor(feature), dim=0)
        return feature
    
