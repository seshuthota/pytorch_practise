import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1, stride=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2, stride=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32,
                                          out_1x1pool=32)
        self.inception3b = InceptionBlock(in_channels=256, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32,
                                          out_5x5=96, out_1x1pool=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = InceptionBlock(in_channels=480, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16,
                                          out_5x5=48, out_1x1pool=64)
        self.inception4b = InceptionBlock(in_channels=512, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24,
                                          out_5x5=64, out_1x1pool=64)
        self.inception4c = InceptionBlock(in_channels=512, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24,
                                          out_5x5=64, out_1x1pool=64)
        self.inception4d = InceptionBlock(in_channels=512, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32,
                                          out_5x5=64, out_1x1pool=64)
        self.inception4e = InceptionBlock(in_channels=528, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32,
                                          out_5x5=128, out_1x1pool=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(in_channels=832, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32,
                                          out_5x5=128, out_1x1pool=128)
        self.inception5b = InceptionBlock(in_channels=832, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48,
                                          out_5x5=128, out_1x1pool=128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, 1000)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


model = GoogLeNet().to(device)

x = torch.randn(1, 3, 224, 224).to(device)

print(model(x).shape)
