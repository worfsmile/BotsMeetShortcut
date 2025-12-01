import torch
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
from transformers.utils import logging

logging.set_verbosity_error()

class HumanValueExtractor:
    def __init__(self, ckpt, device='cuda:0'):
        self.device = device
        self.pipe = pipeline("text-classification", model=ckpt, truncation=True, tokenizer=ckpt, padding=True, max_length=512, top_k=None, device=device)
        self.value2id = {
                          'null': -1,
                          'Self-direction: thought': 0, 
                          'Stimulation': 1, 
                          'Universalism: concern': 2, 
                          'Self-direction: action': 3, 
                          'Benevolence: dependability': 4,
                          'Benevolence: caring': 5, 
                          'Conformity: interpersonal': 6, 
                          'Security: societal': 7, 
                          'Universalism: tolerance': 8, 
                          'Power: dominance': 9, 
                          'Power: resources': 10, 
                          'Universalism: nature': 11, 
                          'Humility': 12, 
                          'Face': 13, 
                          'Conformity: rules': 14, 
                          'Achievement': 15, 
                          'Security: personal': 16, 
                          'Hedonism': 17, 
                          'Tradition': 18}
        self.label_dict = {v: k for k, v in self.value2id.items()}

    def __call__(self, text):  #string -> int
        if text == "" or text is None:
                return -1
        else:
            score = self.pipe(text)[0]
            max_label = self.value2id[max(score, key=lambda x: x['score'])['label']]
            return max_label
