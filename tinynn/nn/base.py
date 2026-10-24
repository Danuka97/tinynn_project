# tinynn/nn/base.py

class MinimalModule:
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def parameters(self):
        return []

    def grads(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)