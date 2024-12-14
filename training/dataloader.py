import json
from training.preprocessing import TextPreprocess

proc = TextPreprocess()

class Dataset:
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        context = item["text"]
        label = item["label"]
        
        return context, label
          