import json
import torch
import numpy as np

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

class DatasetCollator:
    def __init__(self, pretrained_words, seq_length = 200):
        self.seq_length = seq_length
        self.pretrained_words = pretrained_words
    
    def __call__(self, batch):
        contexts, labels = zip(*batch)
        contexts = [self._tokenize(text) for text in contexts]
        return (
            torch.tensor(np.array(contexts), dtype=torch.long),
            torch.tensor(np.array(labels), dtype=torch.float),
        )

    def _tokenize(self, text):
        words = text.split()
        tokenized_text = [self.pretrained_words.get(word, self.pretrained_words['<unk>']) for word in words]
        return self._padding(tokenized_text)

    def _padding(self, tokenized_text):
        features = np.zeros(self.seq_length, dtype=int)
        features[-len(tokenized_text):] = np.array(tokenized_text)[:self.seq_length]
        return features
