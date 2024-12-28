from collections import Counter
from scipy.sparse import csr_matrix

class BoW:
    """customize bag of words"""
    def __init__(self, spare_output=True):
        self.vocab = {}
        self.vocab_size = 0
        self.spare_output = spare_output

    def fit(self, corpus: list):
        vocab_list = set()
        for doc in corpus:
            vocab_list.update(doc.split())
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_list))}
        self.vocab_size = len(self.vocab)

        return self
    
    def transform(self, corpus: list):
        row = []
        for doc in corpus:
            word_count = Counter(doc.split())
            vector = [0] * self.vocab_size
            for word, freq in word_count.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    vector[idx] = freq
            row.append(vector)
        if self.spare_output:
            return csr_matrix(row)
        else:
            return row
    
    def fit_transform(self, corpus: list):
        self.fit(corpus)
        return self.transform(corpus)
    

