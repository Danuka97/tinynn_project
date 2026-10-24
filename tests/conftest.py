"""Shared pytest fixtures."""
import pytest
import torch


@pytest.fixture
def check_grads():
    """Returns a function that checks a module's manual ``backward`` against autograd.

    Runs in float64 so the comparison can use tight tolerances, which means
    modules and inputs must be created inside the test that uses this fixture.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    def check(module, inputs):
        params = module.parameters()

        # Use realistically sized weights: with the default 0.01 init, attention
        # is almost uniform and most gradients are too small for a wrong formula
        # to stand out.
        for p in params:
            if p.dim() == 2:
                p.copy_(torch.randn_like(p) / p.shape[0] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1)

        # Reference gradients from autograd.
        for p in params:
            p.requires_grad_(True)
        x = inputs.clone().requires_grad_(inputs.is_floating_point())
        out = module(x)
        upstream = torch.randn_like(out)
        (out * upstream).sum().backward()
        expected = [p.grad.clone() for p in params]
        for p in params:
            p.grad = None
            p.requires_grad_(False)

        # Manual backward pass.
        module(inputs)
        dx = module.backward(upstream)

        grads = module.grads()
        assert len(grads) == len(expected)
        # atol matters for gradients that are exactly zero in theory (like the
        # attention key bias, which softmax ignores), where both sides are noise.
        for grad, ref in zip(grads, expected):
            assert torch.allclose(grad, ref, rtol=1e-7, atol=1e-10)
        if x.requires_grad:
            assert torch.allclose(dx, x.grad, rtol=1e-7, atol=1e-10)

    yield check
    torch.set_default_dtype(default_dtype)
