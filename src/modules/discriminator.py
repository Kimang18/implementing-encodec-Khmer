from dataclasses import dataclass, field
from typing import Any

import einops
import torch
import torchaudio
from conv import NormConv2d
from torch import nn


@dataclass
class DiscriminatorConfig:
    in_channels: int = 1
    out_channels: int = 1

    filters: int = 32
    max_filters: int = 1024
    filter_scale: int = 1

    # Multiscale STFT Discriminator #
    use_multiscale_freq_discrim: bool = True
    n_ffts: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    hop_lengths: tuple[int, ...] = (512, 256, 128, 64, 32)
    win_lengths: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    kernel_size: tuple[int, ...] = (3, 9)
    dilations: tuple[int, ...] = (1, 2, 4)
    stride: tuple[int, ...] = (1, 2)
    normalized: bool = True
    norm_type: str = "weight_norm"
    activation: str = "LeakyReLU"
    activation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"negative_slope": 0.2}
    )


def init_weights(module, mean=0.0, std=0.01):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(module, "conv"):
            module = module.conv
        module.weight.data.normal_(mean, std)


def get_1d_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_2d_padding(kernel_size, dilation=(1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorSTFT(nn.Module):
    def __init__(
        self,
        filters=32,
        in_channels=1,
        out_channels=1,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        max_filters=1024,
        filter_scale=1,
        kernel_size=(3, 9),
        dilations=[1, 2, 4],
        stride=(1, 2),
        normalized=True,
        norm_type="weight_norm",
        activation="LeakyReLU",
        activation_kwargs: dict = {"negative_slope": 0.2},
    ):

        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(nn, activation)(**activation_kwargs)

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )

        self.convs = nn.ModuleList()
        spec_channels = 2 * self.in_channels  # real and imaginary
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
            )
        )

        in_channels = min(filter_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_channels = min((filter_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm_type=norm_type,
                )
            )
            in_channels = out_channels

        out_channels = min(
            (filter_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm_type=norm_type,
            )
        )
        # final conv to out_channels
        self.conv_post = NormConv2d(
            out_channels,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm_type=norm_type,
        )

    def forward(self, x):
        fmap = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = einops.rearrange(z, "b c w t -> b c t w")

        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)

        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT discriminator
    """

    def __init__(self, config):
        super().__init__()
        assert len(config.n_ffts) == len(config.hop_lengths) == len(config.win_lengths)
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    config.filters,
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    n_fft=config.n_ffts[i],
                    win_length=config.win_lengths[i],
                    hop_length=config.hop_lengths[i],
                    max_filters=config.max_filters,
                    filter_scale=config.filter_scale,
                    kernel_size=config.kernel_size,
                    dilations=config.dilations,
                    stride=config.stride,
                    normalized=config.normalized,
                    norm_type=config.norm_type,
                    activation=config.activation,
                    activation_kwargs=config.activation_kwargs,
                )
                for i in range(len(config.n_ffts))
            ]
        )
        self.n_discriminators = len(self.discriminators)

    def forward(self, real, gen):
        real_outs = []
        gen_outs = []
        real_fmaps = []
        gen_fmaps = []
        for discrim in self.discriminators:
            real_out, real_fmap = discrim(real)
            gen_out, gen_fmap = discrim(gen)

            real_outs.append(real_out)
            gen_outs.append(gen_out)
            real_fmaps.append(real_fmap)
            gen_fmaps.append(gen_fmap)

        return real_outs, gen_outs, real_fmaps, gen_fmaps


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stft_discrim = MultiScaleSTFTDiscriminator(config)

    def forward(self, real, gen):
        real_outs = []
        gen_outs = []
        real_fmaps = []
        gen_fmaps = []

        _real_outs, _gen_outs, _real_fmaps, _gen_fmaps = self.stft_discrim(real, gen)
        real_outs.extend(_real_outs)
        gen_outs.extend(_gen_outs)
        real_fmaps.extend(_real_fmaps)
        gen_fmaps.extend(_gen_fmaps)

        return real_outs, gen_outs, real_fmaps, gen_fmaps


if __name__ == "__main__":
    rand = torch.randn(2, 1, 24000 * 3)
    disc = DiscriminatorSTFT()
    disc(rand)
    config = DiscriminatorConfig()
    discg = Discriminator(config)
