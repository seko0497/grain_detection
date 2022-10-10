from fileinput import filename
from re import I
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


class GrainDataset(Dataset):

    def __init__(self, root_dir, num):

        self.root_dir = root_dir
        self.num = num

        self.crop_flip = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

        self.images = [torch.cat((
            transforms.ToTensor()(
                Image.open(f"{self.root_dir}/{i}_intensity.png")
            ),
            transforms.ToTensor()(
                Image.open(f"{self.root_dir}/{i}_depth.png")
            ),
            transforms.ToTensor()(
                Image.open(f"{self.root_dir}/{i}_target.png")
            )[1:]
            ))
            for i in range(1, 11) if i != 2
        ]

    def __len__(self):

        return self.num

    def __getitem__(self, index):

        index = np.random.randint(0, 9)
        if index + 1 == 2:  # TODO: Segment File 2
            index += 1

        images = self.crop_flip(self.images[index])

        return {"I": images[:2], "O": images[-1:]}


# DEBUG
# grain_dataset = GrainDataset("data/grains", 10)
# grain_dataloader = DataLoader(grain_dataset, batch_size=1)
# for batch in grain_dataloader:
#     __, ax = plt.subplots(3)
#     ax[0].imshow(batch["I"][0][0], cmap="gray")
#     ax[1].imshow(batch["I"][0][1], cmap="gray")
#     ax[2].imshow(batch["O"][0], cmap="gray")
#     plt.show()
