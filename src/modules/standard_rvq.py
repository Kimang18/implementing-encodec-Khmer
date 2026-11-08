import einops
import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, codebook_dim: int | None = None):
        super().__init__()
        codebook_dim = codebook_dim or dim
        requires_projection = codebook_dim != dim
        if requires_projection:
            self.proj_in = nn.Linear(dim, codebook_dim)
            self.proj_out = nn.Linear(codebook_dim, dim)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        self.codebook = nn.Parameter(torch.Tensor(codebook_size, codebook_dim))
        nn.init.uniform_(self.codebook, -1 / codebook_size, 1 / codebook_size)

    def get_codebook_entry(self, code):
        # after retrieve the codebook entry, project it to dim dimension
        z_q = F.embedding(code, self.codebook)
        return self.proj_out(z_q)

    def code_lookup(self, z):
        # z: (batch_size, dim)
        z_e = F.normalize(self.proj_in(z))  # (batch_size, codebook_dim)
        q = F.normalize(self.codebook).t()  # (codebook_dim, codebook_size)
        dist = -1 * (
            z_e.pow(2).sum(dim=1, keepdim=True)
            + q.pow(2).sum(dim=0, keepdim=True)
            - 2 * z_e @ q
        )
        return dist.max(dim=-1).indices

    def forward(self, z):
        code = self.code_lookup(z)
        quantized = self.get_codebook_entry(code)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        return quantized, code, codebook_loss, commitment_loss


class ResidualVectorQuantization(nn.Module):
    def __init__(
        self,
        n_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(dim, codebook_size, codebook_dim)
                for _ in range(n_quantizers)
            ]
        )

    @torch.no_grad()
    def encode(self, x, n_q: int | None = None):
        batch_size, channels, seq_len = x.shape
        residual = einops.rearrange(x, "b c t -> (b t) c")
        codes = []

        n_q = n_q or len(self.quantizers)
        for quantizer in self.quantizers[:n_q]:
            code = quantizer.code_lookup(residual)
            quantized = quantizer.get_codebook_entry(code)
            residual = residual - quantized
            codes.append(code)

        out_codes = torch.stack(codes)  # (n_q, batch_size * seq_len)
        out_codes = einops.rearrange(
            out_codes, "q (b t) -> q b t", b=batch_size, t=seq_len
        )
        return out_codes

    @torch.no_grad()
    def decode(self, codes):
        # q_indices: (n_q, batch_size, seq_len)
        quantized_out = torch.tensor(0.0, device=codes.device)

        for i, code in enumerate(codes):
            quantizer = self.quantizers[i]
            quantized = quantizer.get_codebook_entry(code)
            quantized_out = quantized_out + quantized

        quantized_out = einops.rearrange(quantized_out, "b t c -> b c t")
        return quantized_out

    def forward(self, x, n_q: int | None = None):
        # x: (b c t)
        batch_size, channels, seq_len = x.shape
        x_e = einops.rearrange(x, "b c t -> (b t) c")

        quantized = torch.zeros_like(x_e)
        codes = []
        codebook_losses = 0
        commitment_losses = 0
        n_q = n_q or len(self.quantizers)
        for quantizer in self.quantizers[:n_q]:
            x_q, code, codebook_loss, commitment_loss = quantizer(x_e)

            # state through estimator
            x_q = x_e + (x_q - x_e).detach()
            quantized = quantized + x_q

            # compute residual
            x_e = x_e - x_q

            codes.append(code)
            codebook_losses += codebook_loss
            commitment_losses += commitment_loss

        quantized = einops.rearrange(
            quantized, "(b t) c -> b c t", b=batch_size, t=seq_len
        )
        return (
            quantized,
            torch.stack(codes, dim=-1),
            codebook_losses + self.commitment_weight * commitment_losses,
        )


if __name__ == "__main__":
    rvs = ResidualVectorQuantization(8, 16, 32)
    rand = torch.randn(3, 16, 4)
    output = rvs(rand)
    print(output[0].shape, output[1].shape, output[2])
