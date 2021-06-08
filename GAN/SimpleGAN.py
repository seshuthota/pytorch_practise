import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 784
batch_size = 32
epochs = 50

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

dataset = datasets.MNIST(root="../dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optim_disc = optim.Adam(disc.parameters(), lr=lr)
optim_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake/")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ##Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_loss_fake + disc_loss_real) / 2
        disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        optim_disc.step()

        ##Train Generator
        output = disc(fake).view(-1)
        gen_loss = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        gen_loss.backward()
        optim_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = make_grid(fake, normalize=True)
                img_grid_real = make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
