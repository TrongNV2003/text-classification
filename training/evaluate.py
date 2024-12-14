import torch
import numpy as np
import torch.nn as nn
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
        num_correct = 0

        self.model.eval()
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            output = self.model(inputs)

            test_loss = self.loss_fn(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            pred = torch.round(output.squeeze())

            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        test_acc = num_correct/len(self.test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))