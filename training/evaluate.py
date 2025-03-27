import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from training.utils import AverageMeter


class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
    ) -> None:
        self.test_loader = test_loader
        self.BCE_loss = nn.BCELoss()
        self.CE_loss = nn.CrossEntropyLoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

    def test_cnn(self) -> None:
        test_losses = []
        all_labels = []
        all_preds = []
        latencies = []
        num_correct = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                start_time = time.time()

                output = self.model(inputs)

                end_time = time.time()
                latencies.append(end_time - start_time)

                test_loss = self.BCE_loss(output.squeeze(), labels.float())
                test_losses.append(test_loss.item())

                pred = torch.round(output.squeeze())

                all_labels.extend(labels.cpu().detach().numpy())
                all_preds.extend(pred.cpu().detach().numpy())

                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.cpu().detach().numpy())
                num_correct += np.sum(correct)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        calculate_accuracy(all_labels, all_preds)
        print_metrics(all_labels, all_preds)
        calculate_latency(latencies)

    def test_rnn(self) -> None:
        test_losses = []
        all_labels = []
        all_preds = []
        latencies = []
        num_correct = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                h = self.model.init_hidden(inputs.size(0))
                h = tuple([each.data for each in h])

                start_time = time.time()

                output, h = self.model(inputs, h)

                end_time = time.time()
                latencies.append(end_time - start_time)

                test_loss = self.BCE_loss(output.squeeze(), labels.float())
                test_losses.append(test_loss.item())

                pred = torch.round(output.squeeze())

                all_labels.extend(labels.cpu().detach().numpy())
                all_preds.extend(pred.cpu().detach().numpy())

                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.cpu().detach().numpy())
                num_correct += np.sum(correct)

        print(f"Test loss: {np.mean(test_losses):.3f}")

        calculate_accuracy(all_labels, all_preds)
        print_metrics(all_labels, all_preds)
        calculate_latency(latencies)

    def test_lstm(self) -> None:
        self.model.eval()
        test_losses = AverageMeter()
        latencies = []
        all_labels = []
        all_preds = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                start_time = time.time()
                output = self.model(inputs)
                end_time = time.time()

                loss = self.loss_fn(output.squeeze(), labels.float())
                test_losses.update(loss.item(), inputs.size(0))

                preds = (output.squeeze() >= 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                latency = end_time - start_time
                latencies.append(latency)

        calculate_accuracy(all_labels, all_preds)
        print_metrics(all_labels, all_preds)
        calculate_latency(latencies)

    def test_llm(self):
        self.model.eval()
        latencies = []
        all_labels = []
        all_preds = []
        total_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                start_time = time.time()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                latencies.append(time.time() - start_time)

                loss = self.CE_loss(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        print_metrics(all_labels, all_preds)
        calculate_latency(latencies)
        calculate_accuracy(all_labels, all_preds)


@staticmethod
def print_metrics(true_labels: list, predicted_labels: list) -> None:
    precision = precision_score(
        true_labels, predicted_labels, average="binary"
    )
    recall = recall_score(true_labels, predicted_labels, average="binary")
    f1 = f1_score(true_labels, predicted_labels, average="binary")
    print(f"Precision: {precision * 100:.2f}")
    print(f"Recall: {recall * 100:.2f}")
    print(f"F1-score: {f1 * 100:.2f}")


@staticmethod
def calculate_accuracy(true_labels: list, predicted_labels: list) -> None:
    correct = 0
    total = len(true_labels)
    for true, pred in zip(true_labels, predicted_labels):
        if true == pred:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}")
    return accuracy


@staticmethod
def calculate_latency(latencies: list) -> None:
    p99_latency = np.percentile(latencies, 99)
    average_latency = np.mean(latencies)
    print(f"P99 Latency: {p99_latency * 1000:.3f} ms")
    print(f"Average: {average_latency * 1000:.3f} ms")
