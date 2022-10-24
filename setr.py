import math
import torch
import torch.nn as nn
import einops
import torchvision
from collections import OrderedDict


class SETR(nn.Module):

    def __init__(
        self,
            num_patches,
            image_size,
            num_channels,
            embedding_size,
            n_encoder_heads,
            n_encoder_layers,
            dim_mlp,
            encoder_type,
            features,
            out_channels,
            decoder_method="PUP",
            n_mla_heads=None):

        super(SETR, self).__init__()

        self.num_patches = num_patches
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.n_encoder_layers = n_encoder_layers
        self.decoder_method = decoder_method

        self.image_seq = ImageSequentializer(
            num_patches,
            image_size,
            num_channels,
            embedding_size
        )

        self.transformer_encoder = Encoder(
            n_encoder_layers,
            embedding_size,
            n_encoder_heads,
            dim_mlp,
            dropout=0.0
        )

        if decoder_method == "PUP":
            self.decoder = DecoderPUP(
                embedding_size,
                out_channels,
                features
            )

        elif decoder_method == "MLA":

            self.layer_idxs = [
                int(i * self.n_encoder_layers / n_mla_heads) - 1
                for i in range(1, n_mla_heads + 1)]

            self.decoder = DecoderMLA(
                embedding_size,
                out_channels,
                n_mla_heads=n_mla_heads
            )

        self.sigmoid = nn.Sigmoid()

        self.load_pretrained()

    def forward(self, x):

        x = self.image_seq(x)
        x, intermediate = self.transformer_encoder(x)

        hh = self.image_size[0] // self.num_patches[0]
        ww = self.image_size[1] // self.num_patches[1]

        if self.decoder_method == "PUP":

            x = einops.rearrange(
                x,
                "(h w) b c -> b c h w",
                h=hh,
                w=ww
            )

        elif self.decoder_method == "MLA":

            x = [
                    einops.rearrange(
                        intermediate[i],
                        "(h w) b c -> b c h w",
                        h=hh,
                        w=ww
                    ) for i in self.layer_idxs
            ]

        x = self.decoder(x)

        return x

    def load_pretrained(self):

        vit_l_16 = torchvision.models.vit_l_16(
            weights=torchvision.models.ViT_L_16_Weights.DEFAULT)

        self.transformer_encoder.load_state_dict(
            vit_l_16.encoder.state_dict(), strict=False
        )


class Encoder(nn.Module):

    def __init__(
            self, n_encoder_layers, embedding_size, n_heads, dim_mlp, dropout):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        layers = OrderedDict()
        for i in range(n_encoder_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                embedding_size,
                n_heads,
                dim_mlp,
                dropout,
            )
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(embedding_size, eps=1e-6)

    def forward(self, inp):

        intermediate_outputs = []

        x = self.dropout(inp)
        for layer in self.layers:
            x = layer(x)
            intermediate_outputs.append(x)

        return self.ln(x), intermediate_outputs


class EncoderBlock(nn.Module):

    def __init__(self, embedding_size, n_heads, dim_mlp, dropout):

        super().__init__()
        self.embedding_size = embedding_size

        self.ln_1 = nn.LayerNorm(embedding_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(embedding_size, eps=1e-6)
        self.mlp = torchvision.ops.MLP(
            embedding_size,
            [dim_mlp, embedding_size],
            inplace=False
        )

    def forward(self, inp):

        x = self.ln_1(inp)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + inp

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


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


class DecoderMLA(nn.Module):

    def __init__(self, embedding_size, out_channels, n_mla_heads=4):

        super().__init__()

        self.conv1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embedding_size, embedding_size // 2, 1),
                nn.BatchNorm2d(embedding_size // 2),
                nn.ReLU(inplace=True)
            )
            for _ in range(n_mla_heads)
        ])

        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    embedding_size // 2, embedding_size // 2, 3, padding=1),
                nn.BatchNorm2d(embedding_size // 2),
                nn.ReLU(inplace=True)
            )
            for _ in range(n_mla_heads)
        ])

        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    embedding_size // 2, embedding_size // 4, 3, padding=1),
                nn.BatchNorm2d(embedding_size // 4),
                nn.ReLU(inplace=True),
                nn.Upsample(
                    scale_factor=4, mode="bilinear", align_corners=True)
            )
            for _ in range(n_mla_heads)
        ])

        self.cls = nn.Sequential(
            nn.Conv2d(
                embedding_size // 4 * n_mla_heads, out_channels, 3, padding=1),
            nn.Upsample(
                    scale_factor=4, mode="bilinear", align_corners=True)
        )

    def forward(self, x):

        aggr = self.conv1[0](x[0])
        outs = [self.conv3[0](self.conv2[0](aggr))]

        for i, encoding in enumerate(x[1:]):

            out_1 = self.conv1[i + 1](encoding)
            aggr = aggr + out_1
            outs.append(self.conv3[i + 1](self.conv2[i + 1](aggr)))

        out = torch.cat(outs, dim=1)
        out = self.cls(out)
        return out
