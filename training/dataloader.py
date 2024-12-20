import json
import torch
import numpy as np
from training.preprocessing import TextPreprocess
from transformers import AutoTokenizer

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
        clean_text = TextPreprocess(context).process_text()
        label = item["label"]
        return clean_text, label

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


class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 0) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        contexts, labels = zip(*batch)
        text = self.tokenizer(
            contexts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        labels = torch.arange(len(batch), dtype=torch.float)
        return text, labels