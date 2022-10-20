import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import einops


class GrainDataset(Dataset):

    def __init__(self, root_dir, num, image_idxs, in_channels, train=True):

        self.root_dir = root_dir
        self.num = num
        self.image_idxs = image_idxs
        self.train = train
        self.in_channels = in_channels

        self.crop_flip = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

        self.images = [torch.cat(
            tuple([
                transforms.ToTensor()(
                    Image.open(f"{self.root_dir}/{i}_{feature}.png")
                ) for feature in in_channels
            ])
            + tuple(
                [transforms.ToTensor()(
                    Image.open(f"{self.root_dir}/{i}_target.png")
                )[1:]]
            ))
            for i in image_idxs if i != 2
        ]

    def __len__(self):

        if self.train:
            return self.num
        else:
            return len(self.image_idxs)

    def __getitem__(self, index):

        if self.train:

            index = np.random.randint(0, len(self.images))

            images = self.crop_flip(self.images[index])

            return {"I": images[:len(self.in_channels)], "O": images[-1:]}

        else:

            image = self.images[index]
            image = image[
                :,
                :image.shape[1] // 256 * 256,
                :image.shape[2] // 256 * 256]

            p1 = image.shape[1] // 256
            p2 = image.shape[2] // 256

            inp = einops.rearrange(
                    image[:len(self.in_channels)],
                    "c (p1 h) (p2 w) -> (p1 p2) c h w",
                    p1=p1,
                    p2=p2
                )

            return {"I": inp, "O": image[-1:]}


# DEBUG
# grain_dataset = GrainDataset("data/grains", 10)
# grain_dataloader = DataLoader(grain_dataset, batch_size=1)
# for batch in grain_dataloader:
#     __, ax = plt.subplots(3)
#     ax[0].imshow(batch["I"][0][0], cmap="gray")
#     ax[1].imshow(batch["I"][0][1], cmap="gray")
#     ax[2].imshow(batch["O"][0], cmap="gray")
#     plt.show()
