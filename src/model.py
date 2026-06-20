import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.act_conv1: torch.Tensor | None = None
        self.act_conv2: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.conv3.register_forward_hook(self._save_activations)
        self.conv3.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradients(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        self.gradients = grad_output[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.act_conv1 = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.act_conv1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)

        x = self.conv3(x)
        self.act_conv2 = F.relu(self.bn3(x))
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop3(x)
        x = self.fc2(x)
        return x
