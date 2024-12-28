import numpy as np
from scipy.sparse import csr_matrix


class OneHot:
    def __init__(self, spare_output=True):
        self.vocab = {}
        self.vocab_size = 0
        self.encode_doc = []
        self.spare_output = spare_output

    def fit(self, corpus: list):
        """
        This function will create vocab from the corpus

        Parameters:
            corpus: list of strings

        Returns:
            update vocab and vocab_size
        """

        unique_words = set()
        for doc in corpus:
            unique_words.update(doc.split())

        self.vocab = {
            word: idx for idx, word in enumerate(sorted(unique_words))
        }
        self.vocab_size = len(self.vocab)

        return self

    def transform(self, corpus: list):
        """
        This function will transform the corpus into a One Hot representation
        Represent the corpus as a matrix of shape (n_samples, vocab_size)

        Word is presented 1 if it exists in the document, else return 0

        Parameters:
            corpus: list of strings

        Returns:
            matrix: sparse matrix or csr_matrix
        """

        matrix = []
        for doc in corpus:
            words = doc.split()
            for word in words:
                one_hot = [0] * self.vocab_size
                if word in self.vocab:
                    idx = self.vocab[word]
                    one_hot[idx] = 1
            matrix.append(one_hot)
        if self.spare_output:
            return csr_matrix(matrix)
        else:
            return matrix

    def fit_transform(self, corpus: list):
        """
        This function will fit the corpus and transform it into One Hot vector
        """

        self.fit(corpus)
        return self.transform(corpus)
