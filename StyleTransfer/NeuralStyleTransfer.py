import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch import optim
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29].to(device)

    def forward(self, x):
        features = []

        for layer_idx, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_idx) in self.chosen_features:
                features.append(x)
        return features


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


IMAGE_SIZE = 356

loader = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

model = VGG().to(device).eval()

original_img = load_image("annahathaway.png")
style_img = load_image("style.jpg")

# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated_img = original_img.clone().requires_grad_(True)

# Hyperparams
TOTAL_STEPS = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_img], lr=learning_rate)

for step in tqdm(range(TOTAL_STEPS)):

    generated_features = model(generated_img)
    original_img_features = model(original_img)
    style_img_features = model(style_img)

    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_img_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # compute  Gram Matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated_img, f"generated_{step}.png")
