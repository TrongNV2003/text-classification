import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

class Tester:
    def __init__(
            self,
            model: torch.nn.Module,
            test_loader: DataLoader,
        ) -> None:
            self.test_loader = test_loader
            self.loss_fn = nn.BCELoss()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(self.device)

    def test(self) -> None:
        test_losses = []
        all_labels = []
        all_preds = []
        latencies = []
        num_correct = 0

        self.model.eval()
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            start_time = time.time()

            output = self.model(inputs)

            end_time = time.time()
            latencies.append(end_time - start_time)

            test_loss = self.loss_fn(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            pred = torch.round(output.squeeze())

            all_labels.extend(labels.cpu().detach().numpy())
            all_preds.extend(pred.cpu().detach().numpy())

            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().detach().numpy())
            num_correct += np.sum(correct)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        test_acc = num_correct / len(self.test_loader.dataset)
        print("Accuracy: {:.3f}".format(test_acc))

        self.f1(all_labels, all_preds)
        self.calculate_latency(latencies)

    def test_rnn(self) -> None:
        test_losses = []
        all_labels = []
        all_preds = []
        latencies = []
        num_correct = 0

        self.model.eval()
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            h = self.model.init_hidden(inputs.size(0))
            h = tuple([each.data for each in h])
            
            start_time = time.time()

            output, h = self.model(inputs, h)

            end_time = time.time()
            latencies.append(end_time - start_time)

            test_loss = self.loss_fn(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            pred = torch.round(output.squeeze())

            all_labels.extend(labels.cpu().detach().numpy())
            all_preds.extend(pred.cpu().detach().numpy())

            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().detach().numpy())
            num_correct += np.sum(correct)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))
        test_acc = num_correct / len(self.test_loader.dataset)
        print("Accuracy: {:.3f}".format(test_acc))

        self.f1(all_labels, all_preds)
        self.calculate_latency(latencies)

    @staticmethod
    def f1(label, predict):
        score = f1_score(label, predict, average='binary')
        print(f"F1-score: {score*100:.3f}")
    
    @staticmethod
    def calculate_latency(latencies):
        p95_latency = np.percentile(latencies, 95)
        average_latency = np.mean(latencies)
        print(f"P95 Latency: {p95_latency*1000:.3f} ms")
        print(f"Average: {average_latency*1000:.3f} ms")