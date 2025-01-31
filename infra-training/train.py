import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Read hyperparameters
hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
with open(hyperparams_path, "r") as f:
    hyperparams = json.load(f)

batch_size = int(hyperparams.get("batch-size", 64))
epochs = int(hyperparams.get("epochs", 5))
lr = float(hyperparams.get("lr", 0.001))

# Dummy dataset and model
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = nn.Linear(10, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

# Save model to SageMaker output directory
model_dir = "/opt/ml/model/"
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
print("Model saved!")
