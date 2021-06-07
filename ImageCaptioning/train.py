import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ImageCaptioning.model import CNNToRNN
from ImageCaptioning.utils import load_checkpoint, save_checkpoint, print_examples
from data_loader import get_loader

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]
    )

    train_loader, dataset = get_loader(
        root_folder="../dataset/flicker8k/images",
        annotation_file="../dataset/flicker8k/captions.txt",
        transform=transform,
        num_workers=2
    )
    torch.backends.cudnn.benchmark = True

    load_model = True
    save_model = True

    # HyperParameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 20

    # Tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initalize model, loss etc.
    model = CNNToRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers).to(
        device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        print_examples(model, device, dataset)
        if save_model and epoch % 5 == 0 and epoch != 0\
                :
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint)
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training Loss", loss.item(), global_step=step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train()
