import numpy as np
import torch
from conv import SConv1d, SConvTranspose1d
from lstm import SLSTM
from torch import nn


class SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_sizes=[3, 1],
        dilations=[1, 1],
        activation="ELU",
        activation_kwargs={"alpha": 1.0},
        norm_type="weight_norm",
        norm_kwargs={},
        pad_mode="reflect",
        compress=2,
        true_skip=True,
    ):
        super().__init__()

        hidden = dim // compress
        act = getattr(nn, activation)

        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if i == 0 else hidden
            out_channels = dim if i == len(kernel_sizes) - 1 else hidden

            block += [
                act(**activation_kwargs),
                SConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                    pad_mode=pad_mode,
                ),
            ]

        self.block = nn.Sequential(*block)

        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(
        self,
        channels=1,
        dimension=128,
        n_filters=32,
        n_residual_layers=1,
        ratios=[8, 5, 4, 2],
        activation="ELU",
        activation_kwargs={"alpha": 1.0},
        norm_type="weight_norm",
        norm_kwargs={},
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        pad_mode="reflect",
        true_skip=False,
        compress=2,
        lstm=2,
        lstm_bidirectional=True,
    ):
        super().__init__()

        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = 1
        model = [
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
                pad_mode=pad_mode,
            )
        ]
        for i, ratio in enumerate(ratios):
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm_type=norm_type,
                        norm_kwargs=norm_kwargs,
                        activation=activation,
                        activation_kwargs=activation_kwargs,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Followed by the downsample #
            model += [
                act(**activation_kwargs),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                    pad_mode=pad_mode,
                ),
            ]

            mult *= 2
        if lstm:
            model += [
                SLSTM(
                    mult * n_filters, num_layers=lstm, bidirectional=lstm_bidirectional
                )
            ]
        model += [
            act(**activation_kwargs),
            SConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    def __init__(
        self,
        channels=1,
        dimension=128,
        n_filters=32,
        n_residual_layers=1,
        ratios=[8, 5, 4, 2],
        activation="ELU",
        activation_kwargs={"alpha": 1.0},
        final_activation=None,
        final_activation_kwargs={},
        norm_type="weight_norm",
        norm_kwargs={},
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        pad_mode="reflect",
        true_skip=False,
        compress=2,
        lstm=2,
        lstm_bidirectional=True,
    ):
        super().__init__()

        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = int(2 ** len(ratios))

        # basically reversed order of encoder
        model = [
            SConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [
                SLSTM(
                    mult * n_filters, num_layers=lstm, bidirectional=lstm_bidirectional
                )
            ]

        for i, ratio in enumerate(ratios):
            # Upsample #
            model += [
                act(**activation_kwargs),
                SConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                ),
            ]

            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm_type=norm_type,
                        norm_kwargs=norm_kwargs,
                        activation=activation,
                        activation_kwargs=activation_kwargs,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        model += [
            act(**activation_kwargs),
            SConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
                pad_mode=pad_mode,
            ),
        ]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_kwargs = final_activation_kwargs or {}
            model += [final_act(**final_activation_kwargs)]

        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z)


if __name__ == "__main__":
    encoder = SEANetEncoder()
    decoder = SEANetDecoder()

    rand = torch.rand(1, 1, 16000)
    print("Input Shape:", rand.shape)
    z = encoder(rand)
    print("Latent Shape:", z.shape)
    y = decoder(z)
    print("Decoded Shape:", y.shape)
    assert y.shape == rand.shape
