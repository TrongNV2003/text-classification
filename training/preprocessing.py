import re
import string
import pandas as pd

class TextPreprocess:
    def __init__(self):
        dict = "stopwords/vietnamese-stopwords.txt"
        self.stopwords_dict = dict
    
    def load_stopwords(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        return stopwords

    def remove_stopwords(self, text):
        stopwords = self.load_stopwords(self.stopwords_dict)
        words = text.split()
        words = [word for word in words if word not in stopwords]
        return " ".join(words)

    def process_text(self, text: str):
        cleaned_text = text.translate(str.maketrans("", "", string.punctuation))
        cleaned_text = cleaned_text.lower()
        return cleaned_text

