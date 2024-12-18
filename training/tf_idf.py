import math
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix

class TfidfVectorize:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf = {}

    def _tf(self, freq_word_counts, total_words):
        return freq_word_counts / total_words

    def _idf(self, N_docs, freq):
        return math.log((1 + N_docs) / (1 + freq)) + 1

    def fit(self, corpus):
        """Học vocab và tính IDF cho mỗi từ trong tập corpus"""
        N_docs = len(corpus)
        doc_freq = Counter()

        for doc in corpus:
            unique_words = set(doc.split())
            for word in unique_words:
                doc_freq[word] += 1

        for idx, word in enumerate(doc_freq):
            self.vocabulary_[word] = idx
            self.idf[word] = self._idf(N_docs, doc_freq[word])

    def transform(self, corpus):
        """Convert văn bản thành ma trận TF-IDF"""
        tfidf_matrix = []

        for doc in corpus:
            word_counts = Counter(doc.split())
            total_words = sum(word_counts.values())
            tfidf_vector = np.zeros(len(self.vocabulary_))

            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    tf = self._tf(count, total_words)
                    idf = self.idf[word]
                    tfidf_vector[self.vocabulary_[word]] = tf * idf
                    
            tfidf_matrix.append(tfidf_vector)

        return csr_matrix(tfidf_matrix)

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
