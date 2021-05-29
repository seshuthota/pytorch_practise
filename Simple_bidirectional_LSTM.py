import torch
from torch import nn, optim

# Set device
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
sequence_length = 28
learning_rate = 0.005
batch_size = 64
num_epochs = 1


class Bi_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Bi_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = Bi_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # calculating predictions and loss
        scores = model(data)
        loss = criterion(scores, targets)

        # updating weights
        optimizer.zero_grad()
        loss.backward()

        # Taking step in the direction
        optimizer.step()


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
            x = x.to(device=device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct = (predictions == y).sum()
            num_records = x.shape[0]
    model.train()
    return num_correct / float(num_records)


print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")
