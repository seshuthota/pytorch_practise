import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
INPUT_SIZE = 28
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
NUM_CLASSES = 10
IN_CHANNELS = 1
BATCH_SIZE = 64
LOAD_MODEL = True


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def save_checkpoint(state, filename="model_save_checkpoint.pth.tar"):
    print("=> Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNN(IN_CHANNELS, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if LOAD_MODEL:
    load_checkpoint(torch.load("model_save_checkpoint.pth.tar"), model, optimizer)

for epoch in range(NUM_EPOCHS):
    if epoch % 2 == 0:
        checkpoint_dict = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint_dict)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    print(f"Training Epoch : {epoch}")


def check_accuracy(data_loader, model):
    if data_loader.dataset.train:
        print("Checking accuracy of Train")
    else:
        print("Checking accuracy of test")
    num_correct = 0
    num_records = 0

    # Setting model state to evaluation mode
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device=device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct = (predictions == y).sum()
            num_records = x.shape[0]
    model.train()
    return num_correct / float(num_records)


print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")
