## [Task 2-1]
import time
import copy
import torch
import torchvision
import random

from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

import numpy as np
import matplotlib.pyplot as plt


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

# Reuse the BN version of the MLP model from Task 0-1.
class MLP_BN(nn.Module):  # MLP model with BN between FC layers
    def __init__(self):
        super(MLP_BN, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.bn1 = torch.nn.BatchNorm1d(400)

        self.fc2 = nn.Linear(400, 200)
        self.bn2 = torch.nn.BatchNorm1d(200)

        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1*28*28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x


# Implement MLP with BN and checkpoints (Same as Task 0-1 but with checkpoint blocks)
class CheckpointMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the same MLP architecture as MLP_BN from Task 0-1.
        self.fc1 = nn.Linear(784, 400)
        self.bn1 = torch.nn.BatchNorm1d(400)

        self.fc2 = nn.Linear(400, 200)
        self.bn2 = torch.nn.BatchNorm1d(200)

        self.fc3 = nn.Linear(200, 10)


    def block1(self, x):
        # Implement the first checkpointed block:
        # pass checkpoint through fc1 -> bn1 -> ReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        return x

    def block2(self, x):
        # Implement the second checkpointed block:
        # pass second checkpoint through fc2 -> bn2 -> ReLU
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Apply activation checkpointing to block1 and block2 separately.
        # Use this form:
        #   x = checkpoint(, , use_reentrant=False)
        # After that, apply fc3 without checkpointing.
        x = torch.utils.checkpoint.checkpoint(self.block1, x, use_reentrant=False) #checkpoints block1, recomputes intermediate activations
        x = torch.utils.checkpoint.checkpoint(self.block2, x, use_reentrant=False) #checkpoints block2, recomputes intermediate activations
        x = self.fc3(x)

        return x

# Hyperparameters

BATCH_SIZE = 64
LR = 0.01
EPOCHS = 15

# Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

# Train / Eval

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train_and_measure(model, train_loader, test_loader, epochs, lr):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_accs = []
    epoch_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_time = time.time() - start_time
        test_acc = evaluate(model, test_loader)

        epoch_times.append(epoch_time)
        epoch_accs.append(test_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    return epoch_accs, epoch_times, peak_memory_mb


# Main
mlp_plain = MLP_BN()
mlp_ckpt = CheckpointMLP()

plain_accs, plain_times, plain_peak_mem = train_and_measure(
    mlp_plain, train_loader, test_loader, epochs=EPOCHS, lr=LR
)

ckpt_accs, ckpt_times, ckpt_peak_mem = train_and_measure(
    mlp_ckpt, train_loader, test_loader, epochs=EPOCHS, lr=LR
)

epochs = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, plain_accs, label='MLP w/o Checkpoint')
plt.plot(epochs, ckpt_accs, label='MLP with Checkpoint')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Task 2-1: Test Accuracy vs Epoch")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, plain_times, label='MLP w/o Checkpoint')
plt.plot(epochs, ckpt_times, label='MLP with Checkpoint')
plt.xlabel("Epoch")
plt.ylabel("Training Delay per Epoch (s)")
plt.title("Task 2-1: Training Delay vs Epoch")
plt.grid(True)
plt.legend()
plt.show()

print(f"Peak GPU Memory (MLP w/o checkpoint): {plain_peak_mem:.2f} MB")
print(f"Peak GPU Memory (MLP with checkpoint): {ckpt_peak_mem:.2f} MB")

if plain_peak_mem > 0:
    memory_reduction = 100.0 * (plain_peak_mem - ckpt_peak_mem) / plain_peak_mem
    print(f"Memory reduction: {memory_reduction:.2f}%")


## [Task 2-2]
import time
import copy
import torch
import torchvision
import random

from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Reuse the BN version of the CNN model from Task 0-2.

class CNN_BN(nn.Module): ## CNN model with BN
    def __init__(self):
        super(CNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5)) # num of input channel = 1, num of output channel 32, kernel size = 5×5

        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max-pooling with kernel size = 2×2, stride = 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5) # num of input channel = 1, num of output channels 32, kernel size = 5×5 (5 = 5×5)

        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # The output size of last FC layer must be equal to num of class

        ####################################

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)

        x = self.pool(F.relu(self.bn1(x)))

        x = self.conv2(x)

        x = self.pool(F.relu(self.bn2(x)))

        x = x.view(batchsize,-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        ##################################
        return x

# CNN with BN and optional checkpoints
class CheckpointCNN(nn.Module):
    def __init__(self, checkpoint_mode="none"):
        super().__init__()
        self.checkpoint_mode = checkpoint_mode

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5)) # num of input channel = 1, num of output channel 32, kernel size = 5×5

        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max-pooling with kernel size = 2×2, stride = 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5) # num of input channel = 1, num of output channels 32, kernel size = 5×5 (5 = 5×5)

        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=512, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # The output size of last FC layer must be equal to num of class

        self.bn3 = torch.nn.BatchNorm1d(128)


    def block1(self, x):

        batchsize = x.size(0) #Extract batch size from Input Tensor

        x = self.conv1(x) # Pass input tensor through 1st convolution layer

        x = self.pool(F.relu(self.bn1(x))) # bn1 -> relu -> pool

        return x

    def block2(self, x):

        batchsize = x.size(0) # Also extracts batch size from input tensor

        x = self.conv2(x) # passes input tensor through 2nd convolution layer

        x = self.pool(F.relu(self.bn2(x))) # bn2 -> relu -> pool

        return x

    def forward(self, x):


        # checks checkpoint mode
        if(self.checkpoint_mode == "none"):
          # computes block1 and block2 normally
          x = self.block1(x)
          x = self.block2(x)

        elif(self.checkpoint_mode == "block1"):
          # checkponts block1 to save memory
          x = checkpoint(self.block1, x, use_reentrant=False)
          x = self.block2(x)

        elif(self.checkpoint_mode == "both"):
          # checkpoints both blocks to save max memory
          x = checkpoint(self.block1, x, use_reentrant=False)
          x = checkpoint(self.block2, x, use_reentrant=False)

        # Flatten output tensor across all dimensions except batch
        x = torch.flatten(x, 1)

        # Applies final classification, batch, and relu
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


# Hyperparameters
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 15

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

# Train / Eval
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def train_and_measure(model, train_loader, test_loader, epochs, lr):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_accs = []
    epoch_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_time = time.time() - start_time
        test_acc = evaluate(model, test_loader)

        epoch_times.append(epoch_time)
        epoch_accs.append(test_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    return epoch_accs, epoch_times, peak_memory_mb



cnn_plain = CNN_BN()
cnn_ckpt_first = CheckpointCNN(checkpoint_mode="block1")
cnn_ckpt_both = CheckpointCNN(checkpoint_mode="both")

plain_accs, plain_times, plain_peak_mem = train_and_measure(
    cnn_plain, train_loader, test_loader, epochs=EPOCHS, lr=LR
)

first_accs, first_times, first_peak_mem = train_and_measure(
    cnn_ckpt_first, train_loader, test_loader, epochs=EPOCHS, lr=LR
)

both_accs, both_times, both_peak_mem = train_and_measure(
    cnn_ckpt_both, train_loader, test_loader, epochs=EPOCHS, lr=LR
)

epochs = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, plain_accs, label='CNN w/o Checkpoint')
plt.plot(epochs, first_accs, label='Checkpoint 1st Block')
plt.plot(epochs, both_accs, label='Checkpoint Both Blocks')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Task 2-2: Test Accuracy vs Epoch")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, plain_times, label='CNN w/o Checkpoint')
plt.plot(epochs, first_times, label='Checkpoint 1st Block')
plt.plot(epochs, both_times, label='Checkpoint Both Blocks')
plt.xlabel("Epoch")
plt.ylabel("Training Delay per Epoch (s)")
plt.title("Task 2-2: Training Delay vs Epoch")
plt.grid(True)
plt.legend()
plt.show()

print(f"Peak GPU Memory (CNN w/o checkpoint): {plain_peak_mem:.2f} MB")
print(f"Peak GPU Memory (CNN checkpoint first block): {first_peak_mem:.2f} MB")
print(f"Peak GPU Memory (CNN checkpoint both blocks): {both_peak_mem:.2f} MB")

if plain_peak_mem > 0:
    first_reduction = 100.0 * (plain_peak_mem - first_peak_mem) / plain_peak_mem
    both_reduction = 100.0 * (plain_peak_mem - both_peak_mem) / plain_peak_mem
    print(f"Memory reduction (checkpoint first block): {first_reduction:.2f}%")
    print(f"Memory reduction (checkpoint both blocks): {both_reduction:.2f}%")