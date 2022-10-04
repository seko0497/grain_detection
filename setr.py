import math
from tkinter import image_names
from turtle import forward
from requests import patch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from grain_detection.dataset_grain import GrainDataset
import einops
import matplotlib.pyplot as plt
import numpy as np


class SETR(nn.Module):

    def __init__(self):

        super(SETR, self).__init__()
        self.image_seq = ImageSequentializer(
            (16, 16),
            (256, 256),
            2,
            512
        )

    def forward(self, x):

        return self.image_seq(x)


class ImageSequentializer(nn.Module):

    def __init__(self, num_patches, image_size, num_channels, embedding_size):
        super(ImageSequentializer, self).__init__()
        self.num_patches = num_patches
        self.linear = nn.Linear(
            image_size[0] // num_patches[0] *
            image_size[1] // num_patches[0] *
            num_channels,
            embedding_size
        )
        self.pos_embedding = PositionalEncoding(embedding_size)

    def forward(self, x):

        patches = einops.rearrange(
            x,
            "b c (p1 h) (p2 w) -> b c (p1 p2) h w",
            p1=self.num_patches[0],
            p2=self.num_patches[1]
        )

        # # DEBUG
        # fig, ax = plt.subplots(16, 16)
        # for i in range(self.num_patches[0]):
        #     for j in range(self.num_patches[1]):

        #         ax[i, j].imshow(
        #             patches[0, 0, i * self.num_patches[0] + j],
        #             vmin=0,
        #             vmax=1
        #         )
        # plt.show()

        flat_patches = einops.rearrange(
            patches,
            "b c p h w -> p b (c h w)"
        )

        embeddings = self.linear(flat_patches)
        embeddings = self.pos_embedding(embeddings)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


grain_dataset = GrainDataset("data/grains", 10)
grain_dataloader = DataLoader(grain_dataset, batch_size=1)

setr = SETR()

for batch in grain_dataloader:
    intensity = batch["I"]
    setr(intensity)
