class SGD:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers # A list of our custom layers (like LinearLayer)
        self.lr = learning_rate

    def step(self):
        # Iterate through every layer in our network
        for layer in self.layers:
            # We must only update layers that actually have weights to train
            if hasattr(layer, 'weight'):
                # Update the weights and biases using the gradients
                layer.weight -= self.lr * layer.weight_grad
                layer.bias -= self.lr * layer.bias_grad