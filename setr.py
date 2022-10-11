import math
import torch
import torch.nn as nn
import einops


class SETR(nn.Module):

    def __init__(
        self,
            num_patches,
            image_size,
            num_channels,
            embedding_size,
            n_encoder_heads,
            n_encoder_layers,
            features,
            out_channels,
            decoder_method="PUP"):

        super(SETR, self).__init__()

        self.num_patches = num_patches
        self.embedding_size = embedding_size
        self.image_size = image_size

        self.image_seq = ImageSequentializer(
            num_patches,
            image_size,
            num_channels,
            embedding_size
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=n_encoder_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_encoder_layers
        )

        if decoder_method == "PUP":
            self.decoder = DecoderPUP(
                embedding_size,
                out_channels,
                features
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.image_seq(x)
        x = self.transformer_encoder(x)

        hh = self.image_size[0] // self.num_patches[0]
        ww = self.image_size[1] // self.num_patches[1]

        x = einops.rearrange(
            x,
            "(h w) b c -> b c h w",
            h=hh,
            w=ww
        )
        x = self.decoder(x)

        return x


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

        return embeddings


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


class DecoderPUP(nn.Module):

    def __init__(self, in_channels, out_channels, features):

        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x
