"""Numerical checks for tinynn's manual backpropagation.

Gradients produced by each layer's hand-written ``backward`` are validated
against PyTorch autograd, which serves as the ground-truth reference.
"""
import torch

from tinynn.nn.linear import LinearLayer
from tinynn.nn.loss import MSELoss
from tinynn.nn.activation import LeakyReLU, ReLU
from tinynn.nn.attention import SelfAttention
from tinynn.nn.embed import EmbeddingLayer
from tinynn.nn.transformer import TransformerBlock


def test_linear_forward_matches_torch():
    layer = LinearLayer(in_features=4, out_features=3, device="cpu")
    x = torch.randn(2, 5, 4)  # (batch, seq, features)

    out = layer(x)
    expected = x @ layer.weight + layer.bias

    assert out.shape == (2, 5, 3)
    assert torch.allclose(out, expected)


def test_linear_backward_matches_autograd():
    torch.manual_seed(0)
    layer = LinearLayer(in_features=4, out_features=3, device="cpu")
    x = torch.randn(2, 5, 4)

    # Reference gradients from autograd.
    w = layer.weight.clone().requires_grad_(True)
    b = layer.bias.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)
    upstream = torch.randn(2, 5, 3)
    ((x_ref @ w + b) * upstream).sum().backward()

    # Manual backward pass.
    layer(x)
    dx = layer.backward(upstream)

    assert torch.allclose(layer.weight_grad, w.grad, atol=1e-5)
    assert torch.allclose(layer.bias_grad, b.grad, atol=1e-5)
    assert torch.allclose(dx, x_ref.grad, atol=1e-5)


def test_mse_loss_forward_and_backward():
    criterion = MSELoss()
    y_pred = torch.tensor([[1.0], [2.0], [3.0]])
    y_true = torch.tensor([[1.5], [1.0], [2.0]])

    loss = criterion.forward(y_pred, y_true)
    assert torch.isclose(loss, ((y_pred - y_true) ** 2).mean())

    dy = criterion.backward()
    assert torch.allclose(dy, 2.0 * (y_pred - y_true) / y_pred.numel())


def test_mse_loss_backward_matches_autograd_with_several_outputs():
    # forward() averages over all 12 elements, so the gradient must divide by
    # 12, not just by the batch size of 4
    y_pred = torch.randn(4, 3, requires_grad=True)
    y_true = torch.randn(4, 3)
    ((y_pred - y_true) ** 2).mean().backward()

    criterion = MSELoss()
    criterion.forward(y_pred.detach(), y_true)
    assert torch.allclose(criterion.backward(), y_pred.grad)


def test_relu_forward_backward():
    relu = ReLU()
    x = torch.tensor([[-1.0, 2.0], [0.5, -3.0]])

    out = relu(x)
    assert torch.equal(out, torch.tensor([[0.0, 2.0], [0.5, 0.0]]))

    dx = relu.backward(torch.ones_like(x))
    assert torch.equal(dx, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_leaky_relu_backward_matches_autograd(check_grads):
    check_grads(LeakyReLU(negative_slope=0.1), torch.randn(2, 5, 4))


def test_embedding_backward_matches_autograd(check_grads):
    # 10 lookups into 5 rows, so some ids repeat and their gradients must add up
    check_grads(EmbeddingLayer(5, 4, "cpu"), torch.randint(0, 5, (2, 5)))


def test_self_attention_backward_matches_autograd(check_grads):
    check_grads(SelfAttention(8, "cpu"), torch.randn(2, 6, 8))


def test_transformer_block_backward_matches_autograd(check_grads):
    check_grads(TransformerBlock(8, "cpu"), torch.randn(2, 6, 8))
