import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

CONV_NORMALIZATIONS = [
    "none",  # no normalization
    "weight_norm",
    "spectral_norm",  # weight parameterization normalization
    "time_layer_norm",
    "layer_norm",
    "time_group_norm",  # standard normalization
]


class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return x


def apply_parameterization_norm(module, norm_type):
    assert norm_type in CONV_NORMALIZATIONS

    if norm_type == "weight_norm":
        return weight_norm(module)
    elif norm_type == "spectral_norm":
        return spectral_norm(module)
    else:
        return module


def get_norm_module(module, norm_type, **norm_kwargs):
    assert norm_type in CONV_NORMALIZATIONS

    if norm_type == "layer_norm":
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    if norm_type == "time_group_norm":
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


class NormConv1d(nn.Module):
    def __init__(self, *args, norm_type="none", norm_kwargs={}, **kwargs):
        super().__init__()

        self.conv = apply_parameterization_norm(nn.Conv1d(*args, **kwargs), norm_type)
        self.norm = get_norm_module(self.conv, norm_type, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))


class NormConvTranspose1d(nn.Module):
    def __init__(self, *args, norm_type="none", norm_kwargs={}, **kwargs):
        super().__init__()

        self.convtr = apply_parameterization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm_type
        )
        self.norm = get_norm_module(self.convtr, norm_type, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))


class NormConv2d(nn.Module):
    def __init__(self, *args, norm_type="none", norm_kwargs={}, **kwargs):
        super().__init__()

        self.conv = apply_parameterization_norm(nn.Conv2d(*args, **kwargs), norm_type)
        self.norm = get_norm_module(self.conv, norm_type, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.conv(x))


class NormConvTranspose2d(nn.Module):
    def __init__(self, *args, norm_type="none", norm_kwargs={}, **kwargs):
        super().__init__()

        self.convtr = apply_parameterization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm_type
        )
        self.norm = get_norm_module(self.convtr, norm_type, **norm_kwargs)

    def forward(self, x):
        return self.norm(self.convtr(x))


def get_extra_padding_for_conv1d(length, kernel_size, stride, padding_total=0):
    """
    length = 10, kernel_size=3, stride=2, padding_total=2
    n_frames = (10 - 3 + 2) / 2 + 1 = 5.5
    we want complete window at the end, so n_frames should be integer.
    ideal_n_frames = ceil(n_frames) = 6
    ideal_length = (ideal_n_frames - 1) * 2 + (3 - 2) = 11
    the extra padding is ideal_length - length
    """
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x, paddings, mode="zero", value=0):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            # not enough elements for pad-reflect
            # so, add extra 0s, then pad, and finally trim to remove excess pads
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x, paddings):
    # remove pads
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class SConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        norm_type="none",
        norm_kwargs={},
        pad_mode="reflect",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_type = norm_type
        self.norm_kwargs = norm_kwargs
        self.pad_mode = pad_mode

        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x):
        _, _, t = x.shape  # batch, channel, time

        # effective kernel_size considering dilation
        kernel_size = (self.kernel_size - 1) * self.dilation + 1

        # want padding s.t. output = input / stride
        padding_total = kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(
            t, kernel_size, self.stride, padding_total
        )

        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        return self.conv(x)


class SConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        norm_type="none",
        norm_kwargs={},
        pad_mode="reflect",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type

        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x):
        padding_total = self.kernel_size - self.stride
        y = self.convtr(x)

        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        y = unpad1d(y, (padding_left, padding_right))

        return y


if __name__ == "__main__":
    sconv = SConv1d(in_channels=16, out_channels=32, kernel_size=11, stride=2)

    rand = torch.randn(4, 16, 100)
    print(rand.shape)
    conv_out = sconv(rand)
    print(conv_out.shape)

    sconvtr = SConvTranspose1d(
        in_channels=32, out_channels=16, kernel_size=11, stride=2
    )
    convtr_out = sconvtr(conv_out)
    print(convtr_out.shape)
