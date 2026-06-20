import numpy as np
import torch
from PIL import Image
from matplotlib.figure import Figure
from src.model import CNN
from src import gradcam


def test_generate_gradcam_heatmap() -> None:
    model = CNN()
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    model.train()
    out = model(x)
    out[0, 0].backward()
    heatmap = gradcam.generate_gradcam_heatmap(model)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (28, 28)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_heatmap() -> None:
    heatmap = np.random.rand(28, 28).astype(np.float32)
    img = Image.new("L", (28, 28))
    overlay = gradcam.overlay_heatmap(img, heatmap)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (28, 28)


def test_overlay_heatmap_from_tensor() -> None:
    heatmap = np.random.rand(28, 28).astype(np.float32)
    tensor = torch.randn(1, 28, 28)
    overlay = gradcam.overlay_heatmap(tensor, heatmap)
    assert isinstance(overlay, Image.Image)


def test_plot_heatmap_only() -> None:
    heatmap = np.random.rand(28, 28).astype(np.float32)
    fig = gradcam.plot_heatmap_only(heatmap)
    assert isinstance(fig, Figure)
