import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from training.algorithms.bow import BoW
from training.algorithms.one_hot import OneHot
from training.algorithms.tf_idf import Tfidf
from training.utils import AverageMeter


class Vectorizer:
    def __init__(self) -> None:
        self.vectorizer = Tfidf()

    def fit_transform(self, text_set: list) -> torch.Tensor:
        return self.vectorizer.fit_transform(text_set)

    def transform(self, text_set: list) -> torch.Tensor:
        return self.vectorizer.transform(text_set)


class AlgoTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, vector: list, label: list):
        """This function train the machine learning model"""
        return self.model.fit(vector, label)

    def predict(self, vector: list):
        """This function predict the output of the machine learning model"""
        return self.model.predict(vector)


class DNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        learning_rate: float,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        save_dir: str,
    ) -> None:
        self.epochs = epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_acc = -1

    def train_cnn(self) -> None:
        for epoch in range(1, self.epochs + 1):
            train_losses = AverageMeter()

            self.model.train()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for inputs, labels in self.train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.model.zero_grad()
                    output = self.model(inputs)

                    loss = self.loss_fn(output.squeeze(), labels.float())
                    loss.backward()
                    self.optimizer.step()
                    train_losses.update(loss.item(), inputs.size(0))
                    tepoch.set_postfix({"train_loss": train_losses.avg})
                    tepoch.update(1)

            val_losses = AverageMeter()
            val_acc = AverageMeter()
            self.model.eval()
            with torch.no_grad():
                with tqdm(
                    total=len(self.valid_loader), unit="batches"
                ) as tepoch:
                    tepoch.set_description("validation")
                    for inputs, labels in self.valid_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        output = self.model(inputs)
                        val_loss = self.loss_fn(
                            output.squeeze(), labels.float()
                        )
                        val_losses.update(val_loss.item(), labels.size(0))

                        preds = (output.squeeze() >= 0.5).float()
                        accuracy = (preds == labels).float().mean()
                        val_acc.update(accuracy.item(), labels.size(0))

                        tepoch.set_postfix(
                            {
                                "valid_loss": val_losses.avg,
                                "valid_acc": val_acc.avg,
                            }
                        )
                        tepoch.update(1)

            current_acc = val_acc.avg
            if current_acc > self.best_acc:
                print(
                    f"Validation accuracy improved from {self.best_acc:.4f} to {current_acc:.4f}. Saving..."
                )
                self.best_acc = current_acc
                self._save_model(epoch)

            else:
                print("No improvement in val accuracy.")

    def train_rnn(self) -> None:
        clip = 5  # The maximum gradient value to clip at (to prevent exploding gradients).
        for epoch in range(1, self.epochs + 1):
            train_losses = AverageMeter()

            self.model.train()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for inputs, labels in self.train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    h = self.model.init_hidden(inputs.size(0))
                    h = tuple([each.data for each in h])

                    self.model.zero_grad()

                    output, h = self.model(inputs, h)

                    loss = self.loss_fn(output.squeeze(), labels.float())
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip
                    )  # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    self.optimizer.step()
                    train_losses.update(loss.item(), inputs.size(0))
                    tepoch.set_postfix({"train_loss": train_losses.avg})
                    tepoch.update(1)

            val_losses = AverageMeter()
            val_acc = AverageMeter()

            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(
                        self.device
                    )
                    val_h = self.model.init_hidden(inputs.size(0))
                    val_h = tuple([each.data for each in val_h])

                    output, val_h = self.model(inputs, val_h)
                    val_loss = self.loss_fn(output.squeeze(), labels.float())
                    val_losses.update(val_loss.item(), labels.size(0))

                    preds = (output.squeeze() >= 0.5).float()
                    accuracy = (preds == labels).float().mean()
                    val_acc.update(accuracy.item(), labels.size(0))

                    tepoch.set_postfix(
                        {
                            "valid_loss": val_losses.avg,
                            "valid_acc": val_acc.avg,
                        }
                    )
                    tepoch.update(1)

            current_acc = val_acc.avg
            if current_acc > self.best_acc:
                print(
                    f"Validation accuracy improved from {self.best_acc:.4f} to {current_acc:.4f}. Saving..."
                )
                self.best_acc = current_acc
                self._save_model(epoch)

            else:
                print("No improvement in val accuracy.")

    def train_lstm(self) -> None:
        for epoch in range(1, self.epochs + 1):
            train_losses = AverageMeter()

            self.model.train()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for inputs, labels in self.train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.model.zero_grad()
                    output = self.model(inputs)

                    loss = self.loss_fn(output.squeeze(), labels.float())
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()
                    train_losses.update(loss.item(), inputs.size(0))
                    tepoch.set_postfix({"train_loss": train_losses.avg})
                    tepoch.update(1)

            val_losses = AverageMeter()
            val_acc = AverageMeter()
            self.model.eval()
            with torch.no_grad():
                with tqdm(
                    total=len(self.valid_loader), unit="batches"
                ) as tepoch:
                    tepoch.set_description("validation")
                    for inputs, labels in self.valid_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        output = self.model(inputs)
                        val_loss = self.loss_fn(
                            output.squeeze(), labels.float()
                        )
                        val_losses.update(val_loss.item(), labels.size(0))

                        preds = (output.squeeze() >= 0.5).float()
                        accuracy = (preds == labels).float().mean()
                        val_acc.update(accuracy.item(), labels.size(0))

                        tepoch.set_postfix(
                            {
                                "valid_loss": val_losses.avg,
                                "valid_acc": val_acc.avg,
                            }
                        )
                        tepoch.update(1)

            current_acc = val_acc.avg
            if current_acc > self.best_acc:
                print(
                    f"Validation accuracy improved from {self.best_acc:.4f} to {current_acc:.4f}. Saving..."
                )
                self.best_acc = current_acc
                self._save_model(epoch)
            else:
                print("No improvement in val accuracy.")

    def _save_model(self, epoch: int) -> None:
        save_path = os.path.join(
            self.save_dir, f"model_checkpoint_{epoch + 1}.pth"
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

    def load_model(self, load_path: str) -> None:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found {load_path}")

        checkpoint = torch.load(
            load_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {load_path}")


class LlmTrainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        collator_fn=None,
        evaluate_on_accuracy: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
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
            collate_fn=collator_fn,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=collator_fn,
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        self.loss_fn = nn.CrossEntropyLoss()

        num_training_steps = len(self.train_loader) * epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.evaluate_on_accuracy = evaluate_on_accuracy
        self.best_valid_score = 0 if evaluate_on_accuracy else float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_epoch = 0

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = AverageMeter()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    input_ids = data["input_ids"].to(self.device)
                    attention_mask = data["attention_mask"].to(self.device)
                    labels = data["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    loss = self.loss_fn(logits, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                    train_loss.update(loss.item(), input_ids.size(0))
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    tepoch.set_postfix(
                        {"train_loss": train_loss.avg, "lr": current_lr}
                    )
                    tepoch.update(1)

            valid_score = self.evaluate(self.valid_loader)
            improved = False

            if self.evaluate_on_accuracy:
                if (
                    valid_score
                    > self.best_valid_score + self.early_stopping_threshold
                ):
                    print(
                        f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving..."
                    )
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                    print("Saved best model.")
                else:
                    self.early_stopping_counter += 1
                    print(
                        f"No improvement in val accuracy. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}"
                    )

            else:
                if (
                    valid_score
                    < self.best_valid_score - self.early_stopping_threshold
                ):
                    print(
                        f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_score:.4f}. Saving..."
                    )
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    self._save()
                    self.early_stopping_counter = 0
                    improved = True
                    print("Saved best model.")
                else:
                    self.early_stopping_counter += 1
                    print(
                        f"No improvement in validation loss. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}"
                    )

            if improved:
                print(f"Saved best model at epoch {self.best_epoch}.")

            if self.early_stopping_counter >= self.early_stopping_patience:
                print(
                    f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement."
                )
                break

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        all_preds = []
        all_labels = []

        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)
                eval_loss.update(loss.item(), input_ids.size(0))

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

                if self.evaluate_on_accuracy:
                    all_preds_array = np.concatenate(all_preds)
                    all_labels_array = np.concatenate(all_labels)
                    correct = (
                        (
                            torch.tensor(all_preds_array)
                            == torch.tensor(all_labels_array)
                        )
                        .sum()
                        .item()
                    )

                    total = len(all_labels)
                    accuracy = correct / total if total > 0 else 0
                    tepoch.set_postfix(
                        {"valid_loss": eval_loss.avg, "valid_acc": accuracy}
                    )
                else:
                    tepoch.set_postfix({"valid_loss": eval_loss.avg})

                tepoch.update(1)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = np.mean(all_preds == all_labels)
        precision = precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        f1 = f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        logger.info("\n=== Validation Metrics ===")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-score: {f1 * 100:.2f}%")

        return accuracy if self.evaluate_on_accuracy else eval_loss.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
