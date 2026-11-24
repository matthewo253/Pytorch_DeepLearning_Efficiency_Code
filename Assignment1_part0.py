## [Task 0-1]
import torch
import torchvision

from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import time


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

## Define two different MLP models (with and without BN)
class MLP(nn.Module): ## MLP model without BN
    def __init__(self):
        super(MLP, self).__init__()
        # torch.nn.Linear(size of each input sample, size of each output sample)
        self.fc1 = nn.Linear(784, 400) # Input size should be 784 (28*28*1), which is the sample size of MNIST. Output size is the size of first hidden layer(your design parameter).
        self.fc2 = nn.Linear(400, 200) # Input size should be the size of first hidden layer and output size is the size of second hidden layer(your design parameter).
        self.fc3 = nn.Linear(200, 10) # Input size should be the size of second hidden layer and output size should be 10 (the number of classes).

    def forward(self, x):
        x = x.view(-1, 1*28*28) # Change the shape([batch_size, channels, row, col]) to [batch_size, channels*row*col] for fully-connected layer
        x = F.relu(self.fc1(x)) # Use relu function after fully-connected layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Note that softmax function is not applied here.

        return x


class MLP_BN(nn.Module):  # MLP model with BN between FC layers
    def __init__(self):
        super(MLP_BN, self).__init__()
        # Construct the MLP with BN layers
        self.fc1 = nn.Linear(784, 400)
        self.bn1 = torch.nn.BatchNorm1d(400) # batchnorm after connected linear layer

        self.fc2 = nn.Linear(400, 200)
        self.bn2 = torch.nn.BatchNorm1d(200)

        self.fc3 = nn.Linear(200, 10) #dont add a batchnorm here as it is the last output


    def forward(self, x):
        # Implement the forward pass
        x = x.view(-1, 1*28*28)  #changes shape
        x = F.relu(self.bn1(self.fc1(x))) #Use relu after connected layer and connected batchnorm
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

## Implement training and evaluation of these two models

def train(model, optimizer, criterion, dataloader):
    model.train()

    for images, labels in dataloader:

        # Implement one training step for each mini-batch


        # Move data to same device as model
        images = images.to(device)
        labels = labels.to(device)

        # clears the previous iteration
        optimizer.zero_grad()

        # predicts the output for current batch
        outputs = model(images)

        # find the loss between predicted and exact results
        loss = criterion(outputs, labels)

        # Does backpropagation to compute gradient for each backward step
        loss.backward()

        # Updates model parameters using gradient
        optimizer.step()

        ##################################
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:

            # Implement evaluation for each mini-batch

            # Also moves image and labels to the same device as model
            images = images.to(device)
            labels = labels.to(device)

            # predicts outputs for each current batch
            outputs = model(images)

            # gets the total predicted values for each current batch
            _, predicts = torch.max(outputs.data , 1)
            total += labels.size(0)

            # sums the total correct predictions for each batch
            correct += (predicts==labels).sum().item()


            ##################################
    return correct / total

# Hyperparameters

BATCH_SIZE = 64
LR = 0.01
EPOCHS = 15

# Main Guard (Required on Windows when using num_workers > 0)
if __name__ == '__main__':
    # ----------------------------
    # Dataset & Dataloaders
    # ----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


    # Train / Eval (MLP)
    model_plain = MLP().to(device)
    model_bn = MLP_BN().to(device)

    optimizer_plain = optim.SGD(model_plain.parameters(), lr=LR)
    optimizer_bn = optim.SGD(model_bn.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    acc_plain = []
    acc_bn = []

    for epoch in range(EPOCHS):
        train(model_plain, optimizer_plain, criterion, train_loader)
        train(model_bn, optimizer_bn, criterion, train_loader)

        acc_p = evaluate(model_plain, test_loader)
        acc_b = evaluate(model_bn, test_loader)

        acc_plain.append(acc_p)
        acc_bn.append(acc_b)

        print(f"Epoch {epoch+1}/{EPOCHS} | MLP acc: {acc_p:.4f} | MLP_BN acc: {acc_b:.4f}")

    epochs = np.arange(1, EPOCHS + 1)

    plt.plot(epochs, acc_plain, label="MLP (no BN)")
    plt.plot(epochs, acc_bn, label="MLP + BN")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Train / Eval (CNN)
    model_cnn = CNN().to(device)
    model_cnnbn = CNN_BN().to(device)

    optimizer_cnn = optim.SGD(model_cnn.parameters(), lr=LR)
    optimizer_cnnbn = optim.SGD(model_cnnbn.parameters(), lr=LR)

    acc_cnn = []
    acc_cnnbn = []

    for epoch in range(EPOCHS):
        train(model_cnn, optimizer_cnn, criterion, train_loader)
        train(model_cnnbn, optimizer_cnnbn, criterion, train_loader)

        acc1 = evaluate(model_cnn, test_loader)
        acc2 = evaluate(model_cnnbn, test_loader)

        acc_cnn.append(acc1)
        acc_cnnbn.append(acc2)

        print(f"Epoch {epoch+1}/{EPOCHS} | CNN acc: {acc1:.4f} | CNN + BN acc: {acc2:.4f}")

    plt.plot(epochs, acc_cnn, label="CNN (no BN)")
    plt.plot(epochs, acc_cnnbn, label="CNN + BN")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Epoch (CNN)")
    plt.legend()
    plt.grid(True)
    plt.show()