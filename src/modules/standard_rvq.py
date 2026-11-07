import einops
import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_size
        self.codebook = nn.Parameter(torch.Tensor(codebook_size, codebook_dim))
        nn.init.uniform_(self.codebook, -1 / codebook_size, 1 / codebook_size)

    def get_codebook_entry(self, code):
        return F.embedding(code, self.codebook)

    def code_lookup(self, z):
        # z: (batch_size, codebook_dim)
        q = self.codebook.t()  # (codebook_dim, codebook_size)
        dist = -1 * (
            z.pow(2).sum(dim=1, keepdim=True)
            + q.pow(2).sum(dim=0, keepdim=True)
            + 2 * z @ q
        )
        return dist.max(dim=-1).indices

    def forward(self, z):
        code = self.code_lookup(z)
        quantized = self.get_codebook_entry(code)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2, dim=0).sum()
        commit_loss = torch.mean((z - quantized.detach()) ** 2, dim=0).sum()
        return quantized, code, codebook_loss, commit_loss


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
        codebook_dim = codebook_dim or dim
        require_projection = codebook_dim != dim
        if require_projection:  # codebook_dim = dim
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        else:
            self.proj_in = nn.Linear(dim, codebook_dim)
            self.proj_out = nn.Linear(codebook_dim, dim)
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(codebook_size, codebook_dim) for _ in range(n_quantizers)]
        )

    @torch.no_grad()
    def encode(self, x, n_q: int | None = None):
        batch_size, channels, seq_len = x.shape
        residual = einops.rearrange(x, "b d n -> (b n) d")
        codes = []

        n_q = n_q or len(self.quantizers)
        for quantizer in self.quantizers[:n_q]:
            code = quantizer.code_lookup(residual)
            quantize = quantizer.get_codebook_entry(code)
            residual = residual - quantize
            codes.append(code)

        out_codes = torch.stack(codes)  # (n_q, batch_size * seq_len)
        out_codes = out_codes.reshape(
            n_q, batch_size, seq_len
        )  # (n_q, batch_size, seq_len)
        return out_codes

    @torch.no_grad()
    def decode(self, q_indices):
        # q_indices: (n_codebook, batch_size, seq_len)
        quantize_out = torch.tensor(0.0, device=q_indices.device)

        for i, indices in enumerate(q_indices):
            quantizer = self.quantizers[i]
            quantize = quantizer.get_codebook_entry(indices)
            quantize_out = quantize_out + quantize

        quantize_out = self.proj_out(quantize_out)
        quantize_out = einops.rearrange(quantize_out, "b n d -> b d n")
        return quantize_out

    def forward(self, x, n_q: int | None = None):
        # x: (b d n)
        x_e = einops.rearrange(x, "b d n -> (b n) d")
        x_e = self.proj_in(x_e)

        quantized = torch.zeros_like(x_e)
        codes = []
        codebook_losses = 0
        commit_losses = 0
        n_q = n_q or len(self.quantizers)
        for quantizer in self.quantizers[:n_q]:
            x_q, code, codebook_loss, commit_loss = quantizer(x_e)

            # state through estimator
            x_q = x_e + (x_q - x_e).detach()
            quantized = quantized + x_q

            # compute residual
            x_e = x_e - x_q.detach()

            codes.append(code)
            codebook_losses += codebook_loss
            commit_losses += commit_loss

        quantized = self.proj_out(quantized)
        quantized = quantized.reshape(x.shape)
        return (
            quantized,
            torch.stack(codes, dim=-1),
            codebook_losses / n_q + self.commitment_weight * commit_losses / n_q,
        )


if __name__ == "__main__":
    rvs = ResidualVectorQuantization(8, 16, 32)
    rand = torch.randn(3, 16, 4)
    output = rvs(rand)
    print(output[0].shape, output[1].shape, output[2])
