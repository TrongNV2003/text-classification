from collections import Counter

from scipy.sparse import csr_matrix


class BoW:
    """customize bag of words"""

    def __init__(self, spare_output=True):
        self.vocab = {}
        self.vocab_size = 0
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
        This function will transform the corpus into a BoW representation
        Matrix of shape (n_samples, vocab_size)

        Word is presented as frequency in the document

        Parameters:
            corpus: list of strings

        Returns:
            matrix: sparse matrix or csr_matrix
        """

        matrix = []
        for doc in corpus:
            word_count = Counter(doc.split())
            vector = [0] * self.vocab_size
            for word, freq in word_count.items():
                if word in self.vocab:
                    idx = self.vocab[word]
                    vector[idx] = freq
            matrix.append(vector)
        if self.spare_output:
            return csr_matrix(matrix)
        else:
            return matrix

    def fit_transform(self, corpus: list):
        """
        This function will fit the corpus and transform it into BoW vector
        """

        self.fit(corpus)
        return self.transform(corpus)
