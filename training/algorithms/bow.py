from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from scipy.sparse import csr_matrix, vstack

"""customize bag of words"""
class BoW:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, corpus):
        vocab_list = set()
        for text in corpus:
            vocab_list.update(text.split())
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_list))}
        self.vocab_size = len(self.vocab)
        return self
    
    def transform(self, corpus):
        row = []
        for text in corpus:
            words = text.split()
            word_count = Counter(words)
            vector = [0] * self.vocab_size
            for word, freq in word_count.items():
                if word in self.vocab:
                    vector[self.vocab[word]] = freq
            row.append(vector)
        return csr_matrix(row)
    
    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    

