import sys
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from src.model import CNN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def predict_image(image_path: str, model: nn.Module, device: str, invert: bool = False) -> int:
    image = Image.open(image_path).convert("L")
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
    try:
        state = torch.load("MulticlassCNN.pth", map_location=device, weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError:
        log.error("No se encontró 'MulticlassCNN.pth'. Ejecutá 'python train.py' para entrenar el modelo.")
        sys.exit(1)
    except (RuntimeError, KeyError) as e:
        log.error("El archivo 'MulticlassCNN.pth' es incompatible con la arquitectura actual.")
        log.error("Ejecutá 'python train.py' para reentrenar el modelo.")
        log.error(f"Detalle: {e}")
        sys.exit(1)

    ruta_imagen = sys.argv[1] if len(sys.argv) > 1 else "mi_digito.jpg"
    resultado = predict_image(ruta_imagen, model, device, invert=True)
    log.info(f"Dígito reconocido: {resultado}")
