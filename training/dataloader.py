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
        """
        Get the item at the given index

        Returns:
            text: the text of the item
            label: the label of the item
        """

        item = self.data[index]
        context = item["text"]
        clean_text = preprocess.process_text(context)
        label = item["label"]
        return clean_text, label


class DatasetCollator:
    def __init__(self, pretrained_words: str, seq_length=200) -> None:
        self.seq_length = seq_length
        self.pretrained_words = pretrained_words

    def __call__(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the batch of data and convert tokenized data to tensor

        Parameters:
            batch: list of tuple

        Returns:
            contexts: tensor
            labels: tensor
        """

        contexts, labels = zip(*batch)
        contexts = [self._tokenize(text) for text in contexts]
        return (
            torch.tensor(np.array(contexts), dtype=torch.long),
            torch.tensor(np.array(labels), dtype=torch.float),
        )

    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize the given doc into word
        If the word is not in the pretrained_words, return <unk> token

        Parameters:
            text: str

        Returns:
            features: list of int
        """

        words = text.split()
        tokenized_text = [
            self.pretrained_words.get(word, self.pretrained_words["<unk>"])
            for word in words
        ]
        return self._padding(tokenized_text)

    def _padding(self, tokenized_text: list) -> List[int]:
        """
        Padding the tokenized_text to the seq_length

        Parameters:
            tokenized_text: list of int

        Returns:
            features: list of int
        """

        features = np.zeros(self.seq_length, dtype=int)
        features[-len(tokenized_text) :] = np.array(tokenized_text)[
            : self.seq_length
        ]
        return features


class LlmDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list) -> Mapping[str, torch.Tensor]:
        """
        Tokenize the batch of data and convert tokenized data to tensor

        Parameters:
            batch: list of tuple

        Returns:
            input_ids: tensor
            attention_mask: tensor
            label: tensor
        """

        contexts, labels = zip(*batch)

        texts = self.tokenizer(
            contexts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "text_input_ids": texts["input_ids"].squeeze(),
            "text_attention_mask": texts["attention_mask"].squeeze(),
            "label": torch.tensor(np.array(labels), dtype=torch.long),
        }
