# tinynn/optim/adam.py
import torch


class Adam:
    """Adam optimizer for TinyNN's manually computed gradients.

    Works like SGD but keeps a running average of both the gradient
    (first moment) and the squared gradient (second moment) for every
    parameter, giving each weight its own adaptive step size.
    """

    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.layers = layers  # A list of our custom layers or whole models (like LinearLayer or GPT)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0  # Timestep, used for bias correction

        # Moment buffers, created lazily to match each parameter's shape
        self.m = {}  # First moment (mean of gradients)
        self.v = {}  # Second moment (mean of squared gradients)

    def _update_param(self, key, param, grad):
        # 1. Initialize buffers on first use
        if key not in self.m:
            self.m[key] = torch.zeros_like(param)
            self.v[key] = torch.zeros_like(param)

        # 2. Update the biased moment estimates
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2

        # 3. Correct the bias introduced by zero-initialization
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)

        # 4. Take the adaptive step
        # Decoupled weight decay
        param -= self.lr * self.weight_decay * param

        # Adaptive Adam update
        param -= self.lr * m_hat / (
            torch.sqrt(v_hat) + self.epsilon
        )

    def step(self):
        self.t += 1
        # Iterate through every layer in our network
        for i, layer in enumerate(self.layers):
            # parameters() and grads() line up one-to-one and include any nested
            # layers, so this covers embeddings and whole models too. grads() is
            # re-read every step because LinearLayer.backward() replaces its
            # gradient tensors rather than updating them.
            params, grads = layer.parameters(), layer.grads()
            with torch.no_grad():
                for j, (param, grad) in enumerate(zip(params, grads, strict=True)):
                    self._update_param((i, j), param, grad)
