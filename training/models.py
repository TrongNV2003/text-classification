import torch
import torch.nn as nn
from sklearn.svm import SVC
import torch.nn.functional as F
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

class SVM:
    def __init__(self):
        self.seed = 42
        self.model_svm = SVC(random_state = self.seed)

    def train(self, vector, label):
        trainer = self.model_svm.fit(vector, label)
        return trainer
    
    def predict(self, vector):
        predicter = self.model_svm.predict(vector)
        return predicter

class NB:
    def __init__(self):
        self.model_nb = ComplementNB()

    def train(self, vector, label):
        trainer = self.model_nb.fit(vector, label)
        return trainer
        
    def predict(self, vector):
        prediction = self.model_nb.predict(vector)
        return prediction

class LogisRegression:
    def __init__(self):
        self.seed = 42
        self.model_lr = LogisticRegression(random_state=self.seed)

    def train(self, vector, label):
        trainer = self.model_lr.fit(vector, label)
        return trainer
    
    def predict(self, vector):
        prediction = self.model_lr.predict(vector)
        return prediction

class CNN(nn.Module):
    def __init__(self, embed_model, vocab_size, output_size, embedding_dim,
                num_filters=100, kernel_sizes=[3, 4, 5], drop_prob=0.5):

        super(CNN, self).__init__()

        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))
        self.embedding.weight.requires_grad = True

        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k-2,0))
            for k in kernel_sizes]
            # activation Relu
            )

        self.full_connected = nn.Linear(len(kernel_sizes) * num_filters, output_size)

        self.dropout = nn.Dropout(drop_prob)
        self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (batch_size, num_filters, conv_seq_length)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2) # 1D pool conv_seq_length + (batch_size, num_filters)
        return x_max

    def forward(self, x):
        embeds = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        embeds = embeds.unsqueeze(1)
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        logit = self.full_connected(x)

        return self.sigmoid(logit)
