from __future__ import print_function

import copy
import os

from torch import optim
from torchvision.utils import save_image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Params
IMG_SIZE = 512

loader = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ]
)


def image_loader(image_name):
    img = Image.open(image_name).resize((IMG_SIZE, IMG_SIZE), resample=0, box=None)
    img = loader(img).unsqueeze(0)  # Adding 0 as Batch to match input size of the network
    return img.to(device, torch.float)


style_img = image_loader("style.jpg")

content_img = image_loader("annahathaway.png")

#assert style_img.shape == content_img.shape, "Both the images should be of same dimension"

unloader = transforms.ToPILImage()

plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.01)


# plt.figure()
# imshow(style_img, title="Style Image")
#
# plt.figure()
# imshow(content_img, title="Cotent Image")


class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_loss(cnn, normalization_mean, normaliazation_std, style_img, content_img,
                             content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normaliazation_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = "batch_{i}"
        else:
            raise RuntimeError(f"Unrecognized Layer: {layer.__class__.__name__}")
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"Content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"Style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses


input_img = content_img.clone()

plt.figure()
imshow(input_img, title="Input Image")


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print("Building the style transfer model....")
    model, style_losses, content_losses = get_style_model_and_loss(cnn, normalization_mean=normalization_mean,
                                                                   normaliazation_std=normalization_std,
                                                                   style_img=style_img, content_img=content_img)
    print(model)
    optimizer = get_input_optimizer(input_img)
    print("Optimizing..")
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run : {run}")
                print(f"Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img=content_img,
                            style_img=style_img, input_img=
                            input_img)
plt.figure()
imshow(output, title="Output Image")
save_image(output, "FinalOutput.jpg")
plt.ioff()
plt.show()
