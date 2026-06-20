import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from src.model import CNN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BATCH_SIZE = 60
NUM_EPOCHS = 15
LR = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="dataset/", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="dataset/", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = CNN(in_channels=1, num_classes=10).to(device)
log.info(f"Modelo:\n{model}")
log.info(f"Dispositivo: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses: list[float] = []
test_accs: list[float] = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100.0 * correct / total
    test_accs.append(accuracy)
    log.info(f"  Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

log.info(f"\nPrecisión final en test: {accuracy:.2f}%")

try:
    torch.save(model.state_dict(), "MulticlassCNN.pth")
    log.info("Modelo guardado en MulticlassCNN.pth")
except (OSError, PermissionError) as e:
    log.error(f"Error al guardar el modelo: {e}")
    sys.exit(1)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.set_xlabel("Época")
ax1.set_ylabel("Pérdida", color="tab:red")
ax1.plot(range(1, NUM_EPOCHS+1), train_losses, color="tab:red", marker="o", label="Pérdida")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Precisión (%)", color="tab:blue")
ax2.plot(range(1, NUM_EPOCHS+1), test_accs, color="tab:blue", marker="s", label="Precisión")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("Curvas de entrenamiento")
fig.tight_layout()
plt.savefig("assets/curvas_entrenamiento.png", dpi=150)
plt.close(fig)
log.info("Gráfico guardado en assets/curvas_entrenamiento.png")
