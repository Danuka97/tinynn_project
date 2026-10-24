import torch
from tinynn.nn.linear import LinearLayer
from tinynn.nn.loss import MSELoss
from tinynn.optim.sgd import SGD # The SGD class we built earlier

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Dataset
X_data = torch.tensor([[1.0], [2.0], [3.0]], device=device)
Y_true = torch.tensor([[2.0], [4.0], [6.0]], device=device)

# 2. Initialize the components
model = LinearLayer(in_features=1, out_features=1, device=device)
criterion = MSELoss()
optimizer = SGD([model], learning_rate=0.01)

# Number of times to loop through the data
epochs = 100

for epoch in range(epochs):
    # 1. Forward Pass & Loss
    Y_pred = model.forward(X_data)
    loss = criterion.forward(Y_pred, Y_true)

    # 2. Backward Pass (The Calculus)
    model.zero_grad()       # Reset old gradients
    dY = criterion.backward()  # Get the initial error gradient from the loss function
    model.backward(dY)      # Push gradients backward through the linear layer

    # 3. Optimize (Update the weights)
    optimizer.step()

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")