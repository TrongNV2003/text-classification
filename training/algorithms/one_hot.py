from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.sparse import csr_matrix

class one_hot_encoding:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.encode_data = []

    def fit(self, corpus):
        vocab_list = set()
        for text in corpus:
            vocab_list.update(text.split())
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_list))}
        self.vocab_size = len(self.vocab)
        
        for i in range(self.vocab_size):
            one_hot = [0] * self.vocab_size
            one_hot[i] = 1
            self.encode_data.append(one_hot)
        
        return self
    
    def transform(self, corpus):
        row = []
        for text in corpus:
            words = text.split()
            vector = []
            for word in words:
                if word in self.vocab:
                    index = self.vocab[word]
                    vector.append(self.encode_data[index])
            row.append(csr_matrix(vector))
        return row

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)