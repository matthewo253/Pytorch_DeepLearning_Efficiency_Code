## [Task 1] Write your own code here.
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

BATCH_SIZE = 64
LR = 0.01
EPOCHS = 15

# Data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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


class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, rank, freeze_A=False, freeze_base=True):

        super().__init__()
        # Define a LoRA linear layer with:
        # 1) a frozen or trainable base linear layer
        # 2) low-rank matrices A and B
        # 3) optional freezing of A for LoRA-FA
        self.linear = nn.Linear(in_features, out_features) # creates a base layer
        if(freeze_base == True): # if freeze is true then it freezes the base layer parameters
          for param in self.linear.parameters():
            param.requires_grad = False

        self.loraA = torch.nn.Parameter(torch.randn(in_features, rank)) #creates a low rank matrice for a with random weights
        self.loraB = torch.nn.Parameter(torch.zeros(rank, out_features)) # creates a matrice b with zeros

        if(freeze_A == True): #freezes A if freeze_A is true
          self.loraA.requires_grad = False


    def forward(self, x):

        # Implement the LoRA forward pass:
        # 1) compute the base linear output
        # 2) compute the low-rank update using A and B
        # 3) add them together

        baseOutput = self.linear(x) # computes base linear layer output
        loraOutput = (x @ self.loraA @ self.loraB) # computes low rank approximiation

        return baseOutput + loraOutput # returns sum of base and LoRA



class MLP_BN_LORA(nn.Module):

    def __init__(self, rank, lora_fa=False):

        super().__init__()
        # Define the MLP with BN layers from Task 0-1,
        # but replace the first two linear layers with LoRA layers

        # computes with lora layer instead of linear layer and specifies rank
        self.fc1 = LoRALinear(784, 400, rank, freeze_A=lora_fa)
        self.bn1 = torch.nn.BatchNorm1d(400)

        self.fc2 = LoRALinear(400, 200, rank, freeze_A=lora_fa)
        self.bn2 = torch.nn.BatchNorm1d(200)

        self.fc3 = nn.Linear(200, 10)


    def forward(self, x):

        # Implement the forward pass:
        # flatten -> fc1 -> bn1 -> relu -> fc2 -> bn2 -> relu -> fc3

        x = x.view(-1, 1*28*28) # flattens shape
        x = F.relu(self.bn1(self.fc1(x))) # fc1 -> bn1 -> relu
        x = F.relu(self.bn2(self.fc2(x))) # fc2 -> bn2 -> relu
        x = self.fc3(x)
        return x

def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def train_task1(model, optimizer, criterion, dataloader):
    model.train()
    losses_batch = []

    for inputs, labels in dataloader:
        # Implement one training step for each mini-batch.
        inputs = inputs.to(device)
        labels = labels.to(device)

        # clears gradients, computes forward pass, calculates loss, backpropagate, and update weights
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses_batch.append(loss.item())


    return sum(losses_batch) / len(losses_batch)

@torch.no_grad()
def eval_task1(model, dataloader):
    model.eval()
    total = 0
    correct = 0

    for inputs, labels in dataloader:
        # Implement evaluation for each mini-batch.
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

        total += len(inputs)


    return 100 * correct / total

def train_and_evaluate(model, optimizer, EPOCHS, model_name):

    train_loss_epoch = []
    test_acc_epoch = []
    delay_epoch = []

    for epoch in tqdm(range(EPOCHS), desc=model_name):
        start = time.time()
        train_loss_epoch.append(train_task1(model, optimizer, criterion, train_loader))
        test_acc_epoch.append(eval_task1(model, test_loader))
        delay_epoch.append(time.time() - start)

    return train_loss_epoch, test_acc_epoch, delay_epoch

criterion = nn.CrossEntropyLoss()
fc1_original_params = 400 * 784
fc2_original_params = 200 * 400
fc1_max_rank = min(784, 400)
fc2_max_rank = min(400, 200)
shared_max_rank = min(fc1_max_rank, fc2_max_rank)

# Conventional training
conventional_model = MLP_BN().to(device)
conventional_optimizer = optim.SGD(conventional_model.parameters(), lr=0.01)

train_conventional_loss_epoch, test_conventional_acc_epoch, conventional_delay_epoch = train_and_evaluate(
    conventional_model, conventional_optimizer, EPOCHS, "Conventional"
)

# LoRA training
model_lora_8 = MLP_BN_LORA(rank=8, lora_fa=False).to(device)
optimizer_lora_8 = optim.SGD(model_lora_8.parameters(), lr=0.01)
train_lora_8_loss_epoch, test_lora_8_acc_epoch, lora_8_delay_epoch = train_and_evaluate(
    model_lora_8, optimizer_lora_8, EPOCHS, "LoRA r=8"
)

model_lora_160 = MLP_BN_LORA(rank=160, lora_fa=False).to(device)
optimizer_lora_160 = optim.SGD(model_lora_160.parameters(), lr=0.01)
train_lora_160_loss_epoch, test_lora_160_acc_epoch, lora_160_delay_epoch = train_and_evaluate(
    model_lora_160, optimizer_lora_160, EPOCHS, "LoRA r=160"
)

# LoRA-FA training
model_lora_fa_8 = MLP_BN_LORA(rank=8, lora_fa=True).to(device)
optimizer_lora_fa_8 = optim.SGD(model_lora_fa_8.parameters(), lr=0.01)
train_lora_fa_8_loss_epoch, test_lora_fa_8_acc_epoch, lora_fa_8_delay_epoch = train_and_evaluate(
    model_lora_fa_8, optimizer_lora_fa_8, EPOCHS, "LoRA-FA r=8"
)

model_lora_fa_160 = MLP_BN_LORA(rank=160, lora_fa=True).to(device)
optimizer_lora_fa_160 = optim.SGD(model_lora_fa_160.parameters(), lr=0.01)
train_lora_fa_160_loss_epoch, test_lora_fa_160_acc_epoch, lora_fa_160_delay_epoch = train_and_evaluate(
    model_lora_fa_160, optimizer_lora_fa_160, EPOCHS, "LoRA-FA r=160"
)

print("\nTrainable parameter counts in Task 1:")
print(f"Conventional Full MLP: {count_trainable_parameters(conventional_model)}")
print(f"LoRA r=8:             {count_trainable_parameters(model_lora_8)}")
print(f"LoRA r=160:           {count_trainable_parameters(model_lora_160)}")
print(f"LoRA-FA r=8:          {count_trainable_parameters(model_lora_fa_8)}")
print(f"LoRA-FA r=160:        {count_trainable_parameters(model_lora_fa_160)}")

## Plot the test accuracy and delay of different approaches.
figure5 = plt.figure(figsize=(12, 10))

f5 = figure5.add_subplot(221)
d5 = figure5.add_subplot(222)

epochs = range(EPOCHS)

f5.plot(epochs, test_conventional_acc_epoch, label="conventional_model")
f5.plot(epochs, test_lora_8_acc_epoch, label="lora_8_model")
f5.plot(epochs, test_lora_160_acc_epoch, label="lora_160_model")
f5.plot(epochs, test_lora_fa_8_acc_epoch, label="lora_fa_8_model")
f5.plot(epochs, test_lora_fa_160_acc_epoch, label="lora_fa_160_model")
f5.set_title("Test accuracy over epoch")
f5.set_xlabel("Epoch")
f5.set_ylabel("Test Accuracy")
f5.legend()

d5.plot(epochs, conventional_delay_epoch, label="conventional_model")
d5.plot(epochs, lora_8_delay_epoch, label="lora_8_model")
d5.plot(epochs, lora_160_delay_epoch, label="lora_160_model")
d5.plot(epochs, lora_fa_8_delay_epoch, label="lora_fa_8_model")
d5.plot(epochs, lora_fa_160_delay_epoch, label="lora_fa_160_model")
d5.set_title("Delay over epoch")
d5.set_xlabel("Epoch")
d5.set_ylabel("Delay (seconds)")
d5.legend()

plt.show()

print("Average Delay per Method in Task 1:")
print(f"Conventional Full MLP:  {sum(conventional_delay_epoch) / len(conventional_delay_epoch):.4f} s")
print(f"LoRA r=8:               {sum(lora_8_delay_epoch) / len(lora_8_delay_epoch):.4f} s")
print(f"LoRA r=160:             {sum(lora_160_delay_epoch) / len(lora_160_delay_epoch):.4f} s")
print(f"LoRA-FA r=8:            {sum(lora_fa_8_delay_epoch) / len(lora_fa_8_delay_epoch):.4f} s")
print(f"LoRA-FA r=160:          {sum(lora_fa_160_delay_epoch) / len(lora_fa_160_delay_epoch):.4f} s")