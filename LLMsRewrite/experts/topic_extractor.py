from transformers import pipeline

class TopicExtractor:
    def __init__(self, path_to_model, device):
        self.pipe = pipeline("text-classification", model=path_to_model, batch_size=64, device=device, padding=True, truncation=True, max_length=512)
        self.topic2id = {
            "null": -1,
            "arts_&_culture": 0, 
            "business_&_entrepreneurs": 1, 
            "pop_culture": 2, 
            "daily_life": 3, 
            "sports_&_gaming": 4, 
            "science_&_technology": 5
        }
        self.label_dict = {
            -1 : "null",
            0 : "arts_&_culture",
            1 : "business_&_entrepreneurs",
            2 : "pop_culture",
            3 : "daily_life",
            4 : "sports_&_gaming",
            5 : "science_&_technology"
        }

    def __call__(self, text): # string -> int
        if text == "" or text is None:
            return -1
        else:
            topic = self.pipe(text)[0]
            labels = self.topic2id[topic['label']]
            return labels
