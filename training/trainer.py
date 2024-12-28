import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from training.algorithms.bow import BoW
from training.utils import AverageMeter
from training.algorithms.tf_idf import Tfidf
from training.algorithms.one_hot import OneHot

class Vectorizer:
    def __init__(self):
        self.vectorizer = BoW()
    
    def train_vectorizer(self, text_set):
        return self.vectorizer.fit_transform(text_set)
    
    def test_vectorizer(self, text_set):
        return self.vectorizer.transform(text_set)

class Trainer_trad:
    def __init__(self, model, vector, label):
        self.model = model
        self.vector = vector
        self.label = label

    def train(self):
        return self.model.fit(self.vector, self.label)
    
    def predict(self):
        return self.model.predict(self.vector)

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
        save_path = os.path.join(self.save_dir, f"model_checkpoint_{epoch + 1}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load_model(self, load_path: str) -> None:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {load_path}")

class LlmTrainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        collator_fn = None,
        evaluate_on_accuracy: bool = False
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=collator_fn
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loss = AverageMeter()

        # self.loss_fn = F.binary_cross_entropy_with_logits()
        self.loss_fn = nn.CrossEntropyLoss()

        self.evaluate_on_accuracy = evaluate_on_accuracy
        if evaluate_on_accuracy:
            self.best_valid_score = 0
        else:
            self.best_valid_score = float("inf")

    def train(self) -> None:        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    text_input_ids = data["text_input_ids"].to(self.device)
                    text_attention_mask = data["text_attention_mask"].to(self.device)
                    labels = data["label"].to(self.device)

                    outputs = self.model(input_ids=text_input_ids, attention_mask=text_attention_mask)
                    logits = outputs.logits
                    loss = self.loss_fn(logits, labels)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)
                self._save()
                
            valid_loss = self.evaluate(self.valid_loader)
            if valid_loss < self.best_valid_score:
                print(
                    f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving.")
                self.best_valid_score = valid_loss
                self._save()


    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                
                text_input_ids = data["text_input_ids"].to(self.device)
                text_attention_mask = data["text_attention_mask"].to(self.device)
                labels = data["label"].to(self.device)

                outputs = self.model(input_ids=text_input_ids, attention_mask=text_attention_mask)
                logits = outputs.logits

                loss = self.loss_fn(logits, labels)
                eval_loss.update(loss.item(), self.valid_batch_size)
                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)
        return eval_loss.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)