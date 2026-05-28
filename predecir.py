import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image

# Definir la misma arquitectura (debes copiarla exactamente igual)
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Función de predicción (igual que antes)
def predict_image(image_path, model, device, invert=False):
    image = Image.open(image_path).convert('L')
    if invert:
        image = Image.eval(image, lambda x: 255 - x)
    image = image.resize((28, 28))
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN(in_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load("MulticlassCNN.pth", map_location=device))
    
    ruta_imagen = "mi_digito.jpg"   # Cambia por tu imagen
    resultado = predict_image(ruta_imagen, model, device, invert=True)
    print(f"Dígito reconocido: {resultado}")