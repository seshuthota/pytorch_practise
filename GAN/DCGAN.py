import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64


class DiscriminatorGAN(nn.Module):

    def __init__(self, img_channels, features_d):
        super(DiscriminatorGAN, self).__init__()
        self.img_channels = img_channels
        self.features_d = features_d
        self.disc = nn.Sequential(
            nn.Conv2d(self.img_channels, self.features_d, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 8, 2, 1),
            nn.Conv2d(features_d * 8, 1, 4, 2, 1),
            nn.Sigmoid(),

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class GeneratorGAN(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(GeneratorGAN, self).__init__()
        self.gen = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )

    ]
)

disc = DiscriminatorGAN(CHANNELS_IMG, FEATURES_DISC).to(device)
gen = GeneratorGAN(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

dataset = datasets.MNIST(root="../dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE)
optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE)

criterion = nn.BCELoss()

initialize_weights(gen)
initialize_weights(disc)

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in tqdm(range(NUM_EPOCHS)):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc.zero_grad()
        disc_loss.backward()
        optim_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                          Loss D: {disc_loss:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
