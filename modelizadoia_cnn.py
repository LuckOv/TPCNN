import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from modelo_cnn import CNN

# Hiperparámetros
BATCH_SIZE = 60
NUM_EPOCHS = 15
LR = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

# Datos
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="dataset/", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="dataset/", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Modelo
model = CNN(in_channels=1, num_classes=10).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Entrenamiento con métricas por época
train_losses = []
test_accs = []

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

    # Evaluación
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
    print(f"  Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

# Resultados finales
print(f"\nPrecisión final en test: {accuracy:.2f}%")

# Guardar modelo
torch.save(model.state_dict(), "MulticlassCNN.pth")
print("Modelo guardado en MulticlassCNN.pth")

# Gráfico de entrenamiento
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
plt.show()
print("Gráfico guardado en assets/curvas_entrenamiento.png")
