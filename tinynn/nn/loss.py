class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return ((y_pred - y_true) ** 2).mean()

    def backward(self):
        # The derivative: 2 * (Prediction - Target) / N, where N counts every
        # element, since forward() takes the mean over all of them
        N = self.y_pred.numel()
        dY = (2.0 * (self.y_pred - self.y_true)) / N
        return dY