import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from training.tf_idf import TfidfVectorize

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorize()
    
    def train_vectorizer(self, text_set):
        text_vector = self.vectorizer.fit_transform(text_set)
        return text_vector
    
    def test_vectorizer(self, text_set):
        text_vector = self.vectorizer.transform(text_set)
        return text_vector

class Trainer_trad:
    def __init__(self, model, vector, label: None):
        self.model = model
        self.vector = vector
        self.label = label

    def train(self):
        trainer = self.model.fit(self.vector, self.label)
        return trainer
    
    def predict(self):
        predicter = self.model.predict(self.vector)
        return predicter

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
