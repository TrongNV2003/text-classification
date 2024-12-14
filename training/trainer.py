import os
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.svm import SVC
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000)
    
    def train_vectorizer(self, text_set):
        text_vector = self.vectorizer.fit_transform(text_set)
        return text_vector
    
    def test_vectorizer(self, text_set):
        text_vector = self.vectorizer.transform(text_set)
        return text_vector

class Tokenizer:
    def tokenize(self, pretrained_words, texts_split):
        tokenized_texts = []
        for text in texts_split:
            words = text.split()
            tokenized_text = [pretrained_words.get(word, pretrained_words['<unk>']) for word in words]
            tokenized_texts.append(tokenized_text)
        return tokenized_texts

    def padding(self, tokenized_texts, seq_length):
        features = np.zeros((len(tokenized_texts), seq_length), dtype=int)
        for i, row in enumerate(tokenized_texts):
            features[i, -len(row):] = np.array(row)[:seq_length]
        return features

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            epochs: int,
            learning_rate: float,
            train_loader: DataLoader,
            valid_loader: DataLoader,
        ) -> None:
            self.epochs = epochs
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.loss_fn = nn.BCELoss()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self) -> None:
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_losses = []
            
            self.model.train()

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.model.zero_grad()

                output = self.model(inputs)

                loss = self.loss_fn(output.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)


                    output = self.model(inputs)
                    val_loss = self.loss_fn(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

            epoch_time = time.time() - epoch_start_time

            print(f"Epoch: {epoch + 1}/{self.epochs}",
                f"Train Loss: {np.mean(train_losses):.6f}",
                f"Val Loss: {np.mean(val_losses):.6f}",
                f"Time: {epoch_time:.2f}s")

# Các mô hình classification
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
        # if freeze_embeddings:
        #     self.embedding.requires_grad = True

        self.convs_1d = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding=(k-2,0))
            for k in kernel_sizes])

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


    
