import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def train(args):
    # SageMaker downloads data to /opt/ml/input/data/{channel_name}
    data_path = os.path.join(args.data_dir, "training")  

    # Load dataset from SageMaker input directory
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Define a simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images.view(images.shape[0], -1))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    # Save trained model to SageMaker output directory
    model_path = os.path.join(args.model_dir, "model.pth")
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add arguments for SageMaker input/output paths
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")

    args = parser.parse_args()
    train(args)