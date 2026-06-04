import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

        # Atributos para visualización de mapas de activación
        self.act_conv1 = None
        self.act_conv2 = None
        # Atributos para Grad‑CAM
        self.activations = None
        self.gradients = None
        self._hook_registered = False

    def forward(self, x):
        # Primera capa (guardamos activación después de ReLU)
        self.act_conv1 = F.relu(self.conv1(x))
        x = self.pool(self.act_conv1)

        # Segunda capa (guardamos antes de ReLU para Grad‑CAM)
        x = self.conv2(x)
        self.activations = x   # salida antes de ReLU
        self.act_conv2 = F.relu(x)
        x = self.pool(self.act_conv2)

        # Si se solicitó Grad‑CAM, registramos el hook en las activaciones
        if self._hook_registered and self.activations.requires_grad:
            self.activations.register_hook(self._save_gradients)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def _save_gradients(self, grad):
        self.gradients = grad

    def enable_gradcam(self):
        """Activa el registro del hook para capturar gradientes."""
        self._hook_registered = True

    def disable_gradcam(self):
        """Desactiva el hook (para inferencia normal)."""
        self._hook_registered = False
        self.gradients = None