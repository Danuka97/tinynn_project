"""Tests for the SGD and Adam optimizers."""
import torch

from tinynn.nn.activation import ReLU
from tinynn.nn.embed import EmbeddingLayer
from tinynn.nn.linear import LinearLayer
from tinynn.nn.loss import MSELoss
from tinynn.nn.sequential import Sequential
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


def make_nested_model():
    # An embedding (no bias), a ReLU (no parameters) and layers nested inside
    # Sequential: the cases an optimizer looking for .weight/.bias would miss
    return Sequential(EmbeddingLayer(10, 4, "cpu"), LinearLayer(4, 8, "cpu"),
                      ReLU(), LinearLayer(8, 3, "cpu"))


def run_backward(model):
    """One forward/backward pass; returns copies of the params and grads."""
    model(torch.randint(0, 10, (2, 5)))
    model.backward(torch.randn(2, 5, 3))
    return ([p.clone() for p in model.parameters()],
            [g.clone() for g in model.grads()])


def test_sgd_updates_every_parameter_of_a_nested_model():
    torch.manual_seed(0)
    model = make_nested_model()
    params_before, grads = run_backward(model)

    SGD([model], learning_rate=0.1).step()

    for p, before, g in zip(model.parameters(), params_before, grads):
        assert torch.allclose(p, before - 0.1 * g)


def test_adam_updates_every_parameter_of_a_nested_model():
    torch.manual_seed(0)
    model = make_nested_model()
    params_before, grads = run_backward(model)

    Adam([model], learning_rate=0.1).step()

    # On the first step the bias-corrected moments are exactly g and g**2,
    # so Adam moves each weight by lr * g / (|g| + epsilon)
    for p, before, g in zip(model.parameters(), params_before, grads):
        assert torch.allclose(p, before - 0.1 * g / (g.abs() + 1e-8))
