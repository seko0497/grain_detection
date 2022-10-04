from fileinput import filename
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class GrainDataset(Dataset):

    def __init__(self, root_dir, num):

        self.root_dir = root_dir
        self.num = num

        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):

        return self.num

    def __getitem__(self, index):

        index = np.random.randint(0, 9)
        if index + 1 == 2:  # TODO: Segment File 2
            index += 1

        intensity_file = f"{index + 1}_intensity.png"
        intensity = Image.open(f"{self.root_dir}/{intensity_file}")
        intensity = transforms.ToTensor()(intensity)

        depth_file = f"{index + 1}_depth.png"
        depth = Image.open(f"{self.root_dir}/{depth_file}")
        depth = transforms.ToTensor()(depth)

        target_file = f"{index + 1}_target.png"
        target = Image.open(f"{self.root_dir}/{target_file}")
        target = transforms.ToTensor()(target)[1:]

        all = torch.cat((intensity, depth, target))
        cropped = transforms.RandomCrop(
            256
        )(all)

        images = self.flip(cropped)

        return {"I": images[:2], "O": images[-1]}


# DEBUG
grain_dataset = GrainDataset("data/grains", 10)
grain_dataloader = DataLoader(grain_dataset, batch_size=1)
for batch in grain_dataloader:
    __, ax = plt.subplots(3)
    ax[0].imshow(batch["I"][0][0], cmap="gray")
    ax[1].imshow(batch["I"][0][1], cmap="gray")
    ax[2].imshow(batch["O"][0], cmap="gray")
    plt.show()
