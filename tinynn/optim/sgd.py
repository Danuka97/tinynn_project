class SGD:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers # A list of our custom layers or whole models (like LinearLayer or GPT)
        self.lr = learning_rate

    def step(self):
        # Iterate through every layer in our network
        for layer in self.layers:
            # parameters() and grads() line up one-to-one and include any nested
            # layers, so this covers embeddings and whole models too. grads() is
            # re-read every step because LinearLayer.backward() replaces its
            # gradient tensors rather than updating them.
            for param, grad in zip(layer.parameters(), layer.grads(), strict=True):
                # Update the weights and biases using the gradients
                param -= self.lr * grad