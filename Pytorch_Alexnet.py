import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class AlexNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(11, 11), stride=(4, 4))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)
        # self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        # x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = AlexNet().to(device)
print(model)
x = torch.randn(32, 3, 224, 224).to(device)
print(model(x).shape)
