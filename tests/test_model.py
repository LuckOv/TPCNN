import torch
from src.model import CNN


def test_model_output_shape() -> None:
    model = CNN()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)


def test_model_batch_output() -> None:
    model = CNN()
    x = torch.randn(16, 1, 28, 28)
    out = model(x)
    assert out.shape == (16, 10)
    assert not torch.isnan(out).any()


def test_model_gradcam_hooks() -> None:
    model = CNN()
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    model.train()
    out = model(x)
    out[0, 0].backward()
    assert model.activations is not None
    assert model.gradients is not None
    assert model.activations.shape == (1, 64, 7, 7)
    assert model.gradients.shape == (1, 64, 7, 7)
