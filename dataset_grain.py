import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GrainDataset(Dataset):

    def __init__(self, root_dir, image_idxs, in_channels, keys=["I", "O"],
                 one_hot=False):

        self.root_dir = root_dir
        self.image_idxs = image_idxs
        self.in_channels = in_channels
        self.keys = keys
        self.one_hot = one_hot

        self.images = [torch.cat(
            tuple([
                transforms.ToTensor()(
                    Image.open(f"{self.root_dir}/{i}_{feature}.png")
                ) for feature in in_channels
            ])
            + tuple(
                [transforms.ToTensor()(
                    Image.open(f"{self.root_dir}/{i}_target.png")
                )]
            ))
            for i in image_idxs]

    def __len__(self):

        return len(self.image_idxs)

    def __getitem__(self, index):

        image = self.images[index]
        trg = image[-1:]
        trg = torch.round(trg)
        if self.one_hot:
            trg = torch.nn.functional.one_hot(trg[0].long()).float()
            trg = torch.moveaxis(trg, -1, 0)
        return {self.keys[0]: image[:len(self.in_channels)],
                self.keys[1]: trg}


# DEBUG
# grain_dataset = GrainDataset(
#     "grain_generation/samples/youthful-paper-47/epoch510_steps100",
#     [0, 1, 2, 3],
#     ["intensity", "depth"])
# grain_dataloader = DataLoader(grain_dataset, batch_size=1)
# for batch in grain_dataloader:
#     __, ax = plt.subplots(3)
#     ax[0].imshow(batch["I"][0][0], cmap="gray")
#     ax[1].imshow(batch["I"][0][1], cmap="gray")
#     ax[2].imshow(batch["O"][0][0], cmap="gray")
#     plt.show()
