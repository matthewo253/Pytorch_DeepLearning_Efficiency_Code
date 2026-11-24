import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Reproducibility

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

# Hyperparameters

epochs = 3
train_batch_size = 2048   # Intentionally large for Colab T4
lr = 1e-3

# Dataset

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Model

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

criterion = nn.CrossEntropyLoss()


# Train / Eval


def train(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total

# Main

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

try:
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        test_acc = evaluate(model, test_loader)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("\nIntentional CUDA OOM reproduced with LARGE batch size.")
    else:
        raise e



## [Task 3-1]

# Set appropriate value for train_batch_size
train_batch_size = 256

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

try:
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        test_acc = evaluate(model, test_loader)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("\nIntentional CUDA OOM reproduced with LARGE batch size.")
    else:
        raise e



## [Task 3-2]
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Set appropriate value for train_batch_size and accumulation_steps
train_batch_size = 256
accumulation_steps = 8

lr = 1e-3
epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

def train_with_accumulation(model, loader, optimizer, epoch, accumulation_steps):
    model.train()
    total_loss = 0.0
    total_samples = 0

    # Initialize gradients before starting the epoch.
    # clear any gradients from previous EPOCH
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [GradAccum]", leave=False)

    for step, (images, labels) in enumerate(pbar, start=1):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Scale the loss for gradient accumulation.
        # averages loss across current mini batches
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if step % accumulation_steps == 0:
            # For every accumulation_steps iterations, perform optimizer.step() and reset gradients.
            # updates weights and resets gradients
            optimizer.step()
            optimizer.zero_grad()


        avg_so_far = total_loss / total_samples
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_so_far:.4f}")

    if step % accumulation_steps != 0:
        # Handle the remaining gradients if the number of iterations is not divisible by accumulation_steps.
        # repeats previous step but makes sure everything is properly applied
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

criterion = nn.CrossEntropyLoss()

# Main

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

try:
    for epoch in range(1, epochs + 1):
        train_loss = train_with_accumulation(model, train_loader, optimizer, epoch, accumulation_steps)
        test_acc = evaluate(model, test_loader)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("\nIntentional CUDA OOM reproduced with LARGE batch size.")
    else:
        raise e