import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        embed_model,
        vocab_size,
        output_size,
        embedding_dim,
        num_filters=100,
        kernel_sizes=[3, 4, 5],
        drop_prob=0.5,
    ):

        super(CNN, self).__init__()

        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.from_numpy(embed_model.vectors)
        )
        self.embedding.weight.requires_grad = True

        self.convs_1d = nn.ModuleList(
            [
                nn.Conv2d(
                    1, num_filters, (k, embedding_dim), padding=(k - 2, 0)
                )
                for k in kernel_sizes
            ]
        )

        self.full_connected = nn.Linear(
            len(kernel_sizes) * num_filters, output_size
        )

        self.dropout = nn.Dropout(drop_prob)
        self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(
            3
        )  # (batch_size, num_filters, conv_seq_length)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(
            2
        )  # 1D pool conv_seq_length + (batch_size, num_filters)
        return x_max

    def forward(self, x):
        embeds = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embeds = embeds.unsqueeze(1)
        conv_results = [
            self.conv_and_pool(embeds, conv) for conv in self.convs_1d
        ]
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        logit = self.full_connected(x)

        return self.sigmoid(logit)


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):

        super(RNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(drop_prob)

        self.full_connected = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        # embeddings and lstm_out

        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out[:, -1, :]  # getting the last time step output

        out = self.dropout(lstm_out)
        out = self.full_connected(out)

        sig_out = self.sig(out)

        return sig_out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(device),
        )

        return hidden


class NaiveBayes:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()

    def fit(self, X, y):
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        total_documents = len(y)

        # step 1. Duyệt qua từng văn bản và cập nhật bộ vocab
        for i, (vector, label) in enumerate(zip(X, y)):
            class_counts[label] += 1
            for word_index in vector.nonzero()[
                1
            ]:  # Lấy index của từ trong vector
                word_counts[label][word_index] += 1
            self.vocabulary.update(vector.nonzero()[1])

        # step 2. Tính xác tần số xuất hiện class
        # Ví dụ có 5 đoạn văn, 2 đoạn class 0, 3 đoạn class 1, thì P(0) = 2/5; P(1) = \3/5
        for label in class_counts:
            self.class_probs[label] = class_counts[label] / total_documents

        # step 3. Tính xác suất điều kiện P(x_i | y) của từng từ x_i trong từng class y
        for label in word_counts:
            total_words_in_class = sum(word_counts[label].values())
            for word_index in word_counts[label]:
                self.word_probs[label][word_index] = (
                    word_counts[label][word_index] + 1
                ) / (total_words_in_class + len(self.vocabulary))

    def predict(self, X):
        predictions = []
        for vector in X:
            class_scores = {}
            for label in self.class_probs:
                score = math.log(self.class_probs[label])
                for word_index in vector.nonzero()[1]:
                    score += math.log(
                        self.word_probs[label].get(
                            word_index, 1 / (len(self.vocabulary) + 1)
                        )
                    )
                class_scores[label] = score
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions
