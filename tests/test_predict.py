import torch
from src.model import CNN


def test_prediction_range() -> None:
    model = CNN()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    pred = torch.argmax(out, dim=1).item()
    assert 0 <= pred <= 9


def test_softmax_is_probability() -> None:
    model = CNN()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    probs = torch.softmax(out, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
