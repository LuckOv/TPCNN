import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


def generate_gradcam_heatmap(model: nn.Module) -> np.ndarray:
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


def overlay_heatmap(
    original_image: Image.Image | torch.Tensor | np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.squeeze().cpu().numpy()

    if isinstance(original_image, np.ndarray):
        if original_image.ndim == 3 and original_image.shape[0] == 1:
            original_image = original_image.squeeze(0)
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
        original_image_rgb = np.stack([original_image, original_image, original_image], axis=-1)
    else:
        original_image = np.array(original_image) / 255.0
        original_image_rgb = np.stack([original_image, original_image, original_image], axis=-1)

    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]

    overlay = (1 - alpha) * original_image_rgb + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    return overlay_pil


def plot_heatmap_only(heatmap: np.ndarray, colormap: str = "jet") -> Figure:
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(heatmap, cmap=colormap)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig
