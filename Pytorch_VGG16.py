import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M", ],
}


# After this Flatter 4096x4096x1000


class VGG_net(nn.Module):

    def __init__(self, in_chanels, num_classes, type):
        super(VGG_net, self).__init__()
        self.in_channels = in_chanels
        self.conv_layers = self.create_conv_layers(type)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, int):
                out_channels = x

                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1)),
                    # nn.BatchNorm2d(x),
                    # nn.ReLU()
                ]
                in_channels = x
            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


model = VGG_net(in_chanels=3, num_classes=1000, type=VGG_types["VGG16"]).to(device)
print(model)
img = torch.randn(32, 3, 224, 224).to(device)
print(model(img).shape)
