import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyperparameters
INPUT_SIZE = 784
in_channels = 1
NUM_CLASSES = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 64
NUM_EPOCHS = 10

# device
device = "cuda" if torch.cuda.is_available() else "cpu"


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model = torchvision.models.vgg16(pretrained=True)
print(model.parameters())
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, NUM_CLASSES))
model.to(device)

# data
train_dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    print(f"Training Completed for Epoch : {epoch}")


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train data")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Got accuracy {float(num_correct) / float(num_samples) * 100:.2f}")

    model.train()
    return float(num_correct) / float(num_samples) * 100


check_accuracy(test_loader, model)
check_accuracy(train_loader, model)
