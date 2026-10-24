"""Tests for the GPT model assembled from tinynn modules."""
import torch

from models.gpt import GPT


def make_model(vocab_size=32, block_size=8):
    return GPT(vocab_size=vocab_size, max_seq_len=block_size,
               embed_size=16, num_layers=2, device="cpu")


def test_forward_output_shape():
    model = make_model()
    token_ids = torch.randint(0, 32, (4, 8))  # (batch, seq)

    logits = model(token_ids)
    assert logits.shape == (4, 8, 32)


def test_forward_accepts_shorter_sequences():
    model = make_model()
    token_ids = torch.randint(0, 32, (2, 5))

    logits = model(token_ids)
    assert logits.shape == (2, 5, 32)


def test_backward_populates_all_grads():
    torch.manual_seed(0)
    model = make_model()
    token_ids = torch.randint(0, 32, (2, 8))

    logits = model(token_ids)
    d_logits = torch.randn_like(logits)
    model.backward(d_logits)

    params = model.parameters()
    grads = model.grads()
    assert len(params) == len(grads)
    for p, g in zip(params, grads):
        assert g.shape == p.shape
        assert torch.isfinite(g).all()


def test_parameter_update_reduces_loss():
    """One SGD step on a fixed batch must decrease cross-entropy."""
    torch.manual_seed(0)
    model = make_model()
    token_ids = torch.randint(0, 32, (2, 8))
    x, y = token_ids[:, :-1], token_ids[:, 1:]

    def loss_fn():
        logits = model(x)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, 32), y.reshape(-1))

    loss_before = loss_fn().item()

    # Analytic gradient of cross-entropy w.r.t. logits.
    logits = model(x)
    flat = logits.reshape(-1, 32)
    probs = torch.softmax(flat, dim=-1)
    probs[torch.arange(flat.shape[0]), y.reshape(-1)] -= 1.0
    probs /= flat.shape[0]
    model.backward(probs.view(logits.shape))

    with torch.no_grad():
        for p, g in zip(model.parameters(), model.grads()):
            p -= 0.1 * g

    assert loss_fn().item() < loss_before
