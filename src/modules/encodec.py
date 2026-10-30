import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .quantizer import ResidualVectorQuantization
from .seanet import SEANetDecoder, SEANetEncoder


@dataclass
class EnCodecConfig:
    # Model Configuration #
    channels: int = 1
    dimension: int = 128
    n_filters: int = 32
    n_residual_layers: int = 1
    ratios: tuple[int, ...] = (8, 5, 4, 2)
    activation: str = "ELU"
    activation_kwargs: dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})
    final_activation: str = None
    final_activation_kwargs: dict[str, Any] = field(default_factory=dict)
    norm_type: str = "weight_norm"
    norm_kwargs: dict[str, Any] = field(default_factory=dict)
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    pad_mode: str = "reflect"
    true_skip: bool = False
    compress: int = 2
    lstm: int = 2
    lstm_bidirectional: bool = True

    # Quantization setup #
    n_quantizers: int = 8
    codebook_dim: int = None
    codebook_size: int = 1024
    decay: float = 0.99
    kmeans_init: bool = True
    kmeans_iters: int = 50
    threshold_ema_dead_code: int = 2
    commit_weight: float = 1


class EnCodecModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_quantizers = config.n_quantizers

        self.encoder = SEANetEncoder(
            channels=config.channels,
            dimension=config.dimension,
            n_filters=config.n_filters,
            n_residual_layers=config.n_residual_layers,
            ratios=config.ratios,
            activation=config.activation,
            activation_kwargs=config.activation_kwargs,
            norm_type=config.norm_type,
            norm_kwargs=config.norm_kwargs,
            kernel_size=config.kernel_size,
            last_kernel_size=config.last_kernel_size,
            residual_kernel_size=config.residual_kernel_size,
            dilation_base=config.dilation_base,
            pad_mode=config.pad_mode,
            true_skip=config.true_skip,
            compress=config.compress,
            lstm=config.lstm,
            lstm_bidirectional=config.lstm_bidirectional,
        )
        self.quantizer = ResidualVectorQuantization(
            n_quantizers=config.n_quantizers,
            dim=self.encoder.dimension,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            decay=config.decay,
            kmeans_init=config.kmeans_init,
            kmeans_iters=config.kmeans_iters,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            commitment_weight=config.commit_weight,
        )
        self.decoder = SEANetDecoder(
            channels=config.channels,
            dimension=config.dimension,
            n_filters=config.n_filters,
            n_residual_layers=config.n_residual_layers,
            ratios=config.ratios,
            activation=config.activation,
            activation_kwargs=config.activation_kwargs,
            final_activation=config.final_activation,
            final_activation_kwargs=config.final_activation_kwargs,
            norm_type=config.norm_type,
            norm_kwargs=config.norm_kwargs,
            kernel_size=config.kernel_size,
            last_kernel_size=config.last_kernel_size,
            residual_kernel_size=config.residual_kernel_size,
            dilation_base=config.dilation_base,
            pad_mode=config.pad_mode,
            true_skip=config.true_skip,
            compress=config.compress,
            lstm=config.lstm,
            lstm_bidirectional=config.lstm_bidirectional,
        )

    def _is_distributed(self):
        return (
            dist.is_available() and dist.is_initialized() and dist.get_world_rank() > 1
        )

    def normalize(self, x):
        batch_size, channels, seq_len = x.shape
        mono = x
        if channels > 1:
            mono = x.mean(dim=1, keepdim=True)

        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
        scale = 1e-8 + volume
        x = x / scale
        return x, scale

    def denormalize(self, x, scale):
        return x * scale

    def forward(self, x):
        batch_size, channels, seq_len = x.shape

        x, scale = self.normalize(x)

        encoded = self.encoder(x)

        n_codebooks_to_use = torch.tensor(
            random.randint(0, self.n_quantizers), device=x.device
        )

        if self._is_distributed():
            dist.broadcast(n_codebooks_to_use, src=0)

        quantize_out, _, out_losses = self.quantizer(encoded, n_codebooks_to_use)

        decoded = self.decoder(quantize_out)

        decoded = self.denormalize(decoded, scale)

        # remove extra-padding due to SConv1d
        decoded = decoded[:, :, :seq_len]

        return {
            "encoder_out": encoded,
            "quantize": quantize_out,
            "quantizer_loss": torch.mean(out_losses),
            "decoded": decoded,
        }

    @torch.no_grad()
    def tokenize(self, x):
        input_len = x.shape[-1]

        x, scale = self.normalize(x)

        encoded = self.encoder(x)

        tokens = self.quantizer.encode(encoded)

        # tokens is returned in (n_codebooks, batch, seq_len) #
        return tokens, scale

    @torch.no_grad()
    def decode(self, indices, scale, max_len=None):
        """indices is (n_codebooks, batch, seq_len)"""

        quantize = self.quantizer.decode(indices)
        decoded = self.decoder(quantize)
        decoded = self.denormalize(decoded, scale)

        if max_len is not None:
            decoded = decoded[:, :, :max_len]

        # final output can only be between -1 and 1 as that is the #
        # value range to save our audio #
        decoded = torch.clamp(decoded, -1.0, 1.0)

        return decoded

    @torch.no_grad()
    def passthrough(self, x):
        """pass in audio and return the final reconstruction"""
        tokens, scale = self.tokenize(x)
        reconstruction = self.decode(tokens, scale, max_len=x.shape[-1])

        return reconstruction


if __name__ == "__main__":
    rand = torch.rand(2, 1, 5)
    config = EnCodecConfig()
    model = EnCodecModel(config)
    out, scale = model.normalize(rand)
    denormed = model.denormalize(out, scale)

    print(rand.mean(), rand.std())
    print(out.mean(), out.std())
    print(denormed.mean(), denormed.std())
