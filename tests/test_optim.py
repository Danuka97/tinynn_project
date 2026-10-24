"""Tests for the SGD and Adam optimizers."""
import torch

from tinynn.nn.linear import LinearLayer
from tinynn.nn.loss import MSELoss
from tinynn.optim.sgd import SGD
from tinynn.optim.adam import Adam

X_DATA = torch.tensor([[1.0], [2.0], [3.0]])
Y_TRUE = torch.tensor([[2.0], [4.0], [6.0]])


def train_linear_regression(optimizer, layer, steps=300):
    criterion = MSELoss()
    losses = []
    for _ in range(steps):
        y_pred = layer(X_DATA)
        losses.append(criterion.forward(y_pred, Y_TRUE).item())
        layer.backward(criterion.backward())
        optimizer.step()
    return losses


def test_sgd_reduces_loss():
    torch.manual_seed(0)
    layer = LinearLayer(in_features=1, out_features=1, device="cpu")
    losses = train_linear_regression(SGD([layer], learning_rate=0.05), layer)
    assert losses[-1] < losses[0] * 0.01


def test_adam_reduces_loss():
    torch.manual_seed(0)
    layer = LinearLayer(in_features=1, out_features=1, device="cpu")
    losses = train_linear_regression(Adam([layer], learning_rate=0.05), layer)
    assert losses[-1] < losses[0] * 0.01


def test_adam_learns_true_weight():
    torch.manual_seed(0)
    layer = LinearLayer(in_features=1, out_features=1, device="cpu")
    train_linear_regression(Adam([layer], learning_rate=0.05), layer, steps=1000)
    assert torch.isclose(layer.weight.squeeze(), torch.tensor(2.0), atol=0.1)
    assert torch.isclose(layer.bias.squeeze(), torch.tensor(0.0), atol=0.1)
