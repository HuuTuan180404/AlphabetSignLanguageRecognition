import torch
import pickle
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score,accuracy_score


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

    data = data.view(data.shape[0], -1)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('Accuracy:', accuracy_score(y_test, y_pred) * 100, '%')

    print(path_dir)
    
    with open(f'{path_dir}MLPClassifier.p', 'wb') as f:
        pickle.dump(model, f)



if __name__ == '__main__':
    data, labels = loadDataset("data/DATASET.pickle")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Cuda is available: True")
        device = torch.device("cuda")

    train(data=data, labels=labels, device=device)
