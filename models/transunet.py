from collections import OrderedDict
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TransUNet(nn.Module):

    def __init__(
            self, embedding_size, n_encoder_heads, n_encoder_layers,
            out_channels, dim_mlp):

        super().__init__()

        self.embeddings = Embeddings(embedding_size)
        self.transformer_encoder = Encoder(
            n_encoder_layers,
            embedding_size,
            n_encoder_heads,
            dim_mlp,
            dropout=0.0
        )

        self.decoder = Decoder(
            embedding_size,
            out_channels
        )

    def forward(self, x):

        x, features = self.embeddings(x)
        x, __ = self.transformer_encoder(x)
        x = self.decoder(x, features)

        return x


class Embeddings(nn.Module):

    def __init__(self, embedding_size, dropout=0.1):

        super().__init__()

        self.embedding_size = embedding_size

        self.resnet = ResNetV2(
            (3, 4, 9),
            1)

        self.patch_embeddings = nn.Conv2d(
            1024, self.embedding_size, kernel_size=1, stride=1)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, 256, self.embedding_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x, features = self.resnet(x)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


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


class Decoder(nn.Module):

    def __init__(self, embedding_size, out_channels):

        super().__init__()

        self.embedding_size = embedding_size

        self.root_conv = nn.Sequential(
            nn.Conv2d(embedding_size, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512 * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256 * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.upsample3_final = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, x, features):

        # x : (B x n_patches x emb_size)

        x = einops.rearrange(  # (B x emb_size x 16 x 16)
                x,
                "b (h w) c -> b c h w",
                h=16,
                w=16)

        x = self.root_conv(x)

        x = self.upsample1(x)  # (B x 512 x 32 x 32)
        x = torch.cat((x, features[0]), dim=1)  # (B x 1024 x 32 x 32)
        x = self.conv1(x)  # (B x 256, 32, 32)

        x = self.upsample2(x)  # (B x 256 x 64 x 64)
        x = torch.cat((x, features[1]), dim=1)  # (B x 512 x 64 x 64)
        x = self.conv2(x)  # (B x 128, 64, 64)

        x = self.upsample3(x)  # (B x 128 x 128 x 128)
        x = torch.cat((x, features[2]), dim=1)  # (B x 192 x 128 x 128)
        x = self.conv3(x)  # (B x 64, 128, 128)

        x = self.upsample3_final(x)  # (B x 128 x 256 x 256)
        x = self.conv_final(x)  # (B x 16 x 256 x 256)

        x = self.cls(x)  # (B x 1 x 256 x 256)

        return x


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(
                2, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*4, cout=width*4, cmid=width)) for i in range(
                        2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*8, cout=width*8, cmid=width*2)) for i in range(
                        2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*16, cout=width*16, cmid=width*4)) for i in range(
                        2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size)
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
