import torch
import pickle
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model.model import MyModel, MLP
from data.utils import CLASSES

def loadDataset(filename: str):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset['data'], dataset['labels']

def train(data, labels, epochs=100, batch_size=64, lr=1e-3, device="cpu"):
    current_time = datetime.now().strftime("%H-%M")
    path_dir = "out-checkpoints/" + current_time + "/"
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    # Convert sang tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val   = torch.tensor(x_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = MLP(input_dim=42, num_classes=len(CLASSES))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        print(f"\t\tEpoch {epoch+1}/{epochs}, Val Accuracy: {correct/total:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{path_dir}model{epoch}.pth")
            print(f"=== Saving model to {best_acc:.2%} ===")

if __name__ == '__main__':
    data, labels = loadDataset("data/DATASET.pickle")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Cuda is available: True")
        device = torch.device("cuda")

    train(data=data, labels=labels, device=device)
