import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from training.tf_idf import TfidfVectorize
import os

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
    def __init__(self, model, vector, label):
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
            save_dir: str
        ) -> None:
            self.epochs = epochs
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.loss_fn = nn.BCELoss()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

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

            self.save_model(epoch)

            print(f"Epoch: {epoch + 1}/{self.epochs}",
                f"Train Loss: {np.mean(train_losses):.6f}",
                f"Val Loss: {np.mean(val_losses):.6f}",
                f"Time: {epoch_time:.2f}s")
            

    def train_rnn(self) -> None:
        clip = 5 # The maximum gradient value to clip at (to prevent exploding gradients).
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_losses = []
            
            self.model.train()

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                h = self.model.init_hidden(inputs.size(0))
                h = tuple([each.data for each in h])

                self.model.zero_grad()

                output, h = self.model(inputs, h)

                loss = self.loss_fn(output.squeeze(), labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip) # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                self.optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    val_h = self.model.init_hidden(inputs.size(0))
                    val_h = tuple([each.data for each in val_h])

                    output, val_h = self.model(inputs, val_h)
                    val_loss = self.loss_fn(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

            epoch_time = time.time() - epoch_start_time

            self.save_model(epoch)

            print(f"Epoch: {epoch + 1}/{self.epochs}",
                f"Train Loss: {np.mean(train_losses):.6f}",
                f"Val Loss: {np.mean(val_losses):.6f}",
                f"Time: {epoch_time:.2f}s")
    
    def save_model(self, epoch: int) -> None:
        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        Tải lại trạng thái của model và optimizer từ file.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Model loaded from {load_path}, starting from epoch {start_epoch}")