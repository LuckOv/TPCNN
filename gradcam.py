import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import io

def generate_gradcam_heatmap(model):
    activations = model.activations.detach()
    gradients = model.gradients.detach()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    for i in range(pooled_gradients.size(0)):
        activations[0, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()

    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    heatmap = np.array(Image.fromarray(heatmap).resize((28, 28)))

    return heatmap


def overlay_heatmap(original_image, heatmap, alpha=0.5, colormap='jet'):
    """
    Superpone el mapa de calor sobre la imagen original.
    Args:
        original_image: Imagen PIL (28x28, escala de grises)
        heatmap: Array numpy del mapa de calor (28x28)
        alpha: Transparencia del heatmap
        colormap: Mapa de colores (jet, viridis, hot, etc.)
    Returns:
        overlay: Imagen PIL con superposición de colores
    """
    # Asegurar formato correcto
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.squeeze().cpu().numpy()

    if isinstance(original_image, np.ndarray):
        if original_image.ndim == 3 and original_image.shape[0] == 1:
            original_image = original_image.squeeze(0)
        # Normalizar imagen original para visualización
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
        # Convertir a RGB (repetir el canal)
        original_image_rgb = np.stack([original_image, original_image, original_image], axis=-1)
    else:
        # Si es imagen PIL, convertir a numpy array
        original_image = np.array(original_image) / 255.0
        original_image_rgb = np.stack([original_image, original_image, original_image], axis=-1)

    # Aplicar colormap al heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Quitar canal alfa

    # Superponer heatmap con transparencia
    overlay = (1 - alpha) * original_image_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    # Convertir a imagen PIL
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    return overlay_pil


def plot_heatmap_only(heatmap, colormap='jet'):
    """
    Genera una figura de matplotlib con solo el mapa de calor.
    Args:
        heatmap: Array numpy del mapa de calor
        colormap: Mapa de colores
    Returns:
        fig: Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(heatmap, cmap=colormap)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig