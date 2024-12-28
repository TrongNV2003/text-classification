import math
from collections import Counter

from scipy.sparse import csr_matrix


class Tfidf:
    def __init__(self, spare_output=True):
        self.idf = {}
        self.vocabulary_ = {}
        self.spare_output = spare_output

    def _tf(self, word_freq: int, total_words: int):
        """
        Calculate TF for a word in a document

        Parameters:
            word_freq: int
            total_words: int

        Returns:
            tf: float
        """

        return word_freq / total_words

    def _idf(self, total_docs: int, doc_freq: int):
        """
        Calculate IDF for a word in corpus

        Parameters:
            total_docs: int
            doc_freq: int

        Returns:
            idf: float
        """

        return math.log((1 + total_docs) / (1 + doc_freq)) + 1

    def fit(self, corpus: list):
        """
        This function will create vocab from the corpus and calculate the IDF for each word in vocab

        Parameters:
            corpus: list of strings

        Returns:
            update vocab and idf
        """

        total_docs = len(corpus)
        doc_freq = Counter()

        for doc in corpus:
            unique_words = set(doc.split())
            for word in unique_words:
                doc_freq[word] += 1

        for idx, word in enumerate(doc_freq):
            self.vocabulary_[word] = idx
            self.idf[word] = self._idf(total_docs, doc_freq[word])

        return self

    def transform(self, corpus: list):
        """
        This function will transform the corpus into a BoW representation


        Parameters:
            corpus: list of strings

        Returns:
            matrix: sparse matrix or csr_matrix
        """

        matrix = []
        for doc in corpus:
            word_counts = Counter(doc.split())
            total_words = sum(word_counts.values())
            tfidf_vector = [0] * len(self.vocabulary_)

            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    tf = self._tf(count, total_words)
                    idf = self.idf[word]
                    idx = self.vocabulary_[word]
                    tfidf_vector[idx] = tf * idf

            matrix.append(tfidf_vector)
        if self.spare_output:
            return csr_matrix(matrix)
        else:
            return matrix

    def fit_transform(self, corpus: list):
        """
        This function will fit the corpus and transform it into Tf-Idf vector
        """

        self.fit(corpus)
        return self.transform(corpus)
