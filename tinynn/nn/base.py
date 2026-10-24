# tinynn/nn/base.py

class MinimalModule:
    def zero_grad(self):
        # Our gradients live in each layer's *_grad buffers, not in p.grad
        # (we never use autograd), so those are what we clear
        for g in self.grads():
            g.zero_()

    def parameters(self):
        return []

    def grads(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)