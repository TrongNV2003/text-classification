import json
from typing import List, Mapping, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from training.preprocessing import TextPreprocess

preprocess = TextPreprocess()


class Dataset:
    def __init__(self, json_file: str) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        item = self.data[index]
        context = item["text"]
        clean_text = preprocess.process_text(context)
        label = item["label"]
        return clean_text, label


class DnnDataCollator:
    def __init__(self, pretrained_words: str, max_length=200) -> None:
        self.max_length = max_length
        self.pretrained_words = pretrained_words

    def __call__(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        contexts, labels = zip(*batch)
        contexts = [self._tokenize(text) for text in contexts]
        contexts_array = np.stack(contexts)
        return (
            torch.tensor(contexts_array, dtype=torch.long),
            torch.tensor(np.array(labels), dtype=torch.float),
        )

    def _tokenize(self, text: str) -> List[int]:
        if not text or not isinstance(
            text, str
        ):  # Handle empty or invalid input
            return self._padding([])

        words = text.split()
        tokenized_text = [
            self.pretrained_words.get(word, self.pretrained_words["<unk>"])
            for word in words
        ]
        return self._padding(tokenized_text)

    def _padding(self, tokenized_text: list) -> List[int]:
        features = np.zeros(self.max_length, dtype=int)
        if not tokenized_text or not isinstance(tokenized_text, list):
            return features.tolist()

        token_len = len(tokenized_text)
        if token_len > self.max_length:
            features[:] = tokenized_text[: self.max_length]
        else:
            features[-token_len:] = tokenized_text
        return features.tolist()


class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        contexts, labels = zip(*batch)

        texts = self.tokenizer(
            contexts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": texts["input_ids"].squeeze(),
            "attention_mask": texts["attention_mask"].squeeze(),
            "labels": torch.tensor(np.array(labels), dtype=torch.long),
        }
