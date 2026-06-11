import torch
import torchvision.transforms as transforms
from PIL import Image
from modelo_cnn import CNN

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