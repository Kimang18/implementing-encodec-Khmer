import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

WORLD_SIZE = dist.get_world_size()
IS_MAIN_PROCESS = dist.get_rank() == 0


def ema_inplace(moving_average, new, decay):
    """moving_average=moving_average * gamma + (1-gamma)*new"""
    moving_average.data.mul_(decay).add_(new, alpha=1 - decay)


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num):
    n_samples = samples.shape[0]
    device = samples.device

    if n_samples > num:
        indices = torch.randperm(n_samples, device=device)[:num]
    else:
        indices = torch.randint(0, n_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, n_clusters, n_iters):
    dim = samples.shape[1]
    dtype = samples.dtype

    centroids = sample_vectors(samples, n_clusters)

    for _ in range(n_iters):
        samples_, centroids_ = samples.unsqueeze(1), centroids.unsqueeze(0)

        diffs = samples_ - centroids_  # (n, k, dim)
        dists = -1 * torch.sum(diffs**2, dim=-1)
        buckets = dists.max(dim=-1).indices  # (n,)

        bins = torch.bincount(buckets, minlength=n_clusters)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill_(zero_mask, 1)  # avoid division by 0

        new_centroids = buckets.new_zeros(n_clusters, dim, dtype=dtype)
        repeated_buckets = buckets.unsqueeze(-1).repeat(1, dim)
        new_centroids.scatter_add_(dim=0, index=repeated_buckets, src=samples)
        new_centroids = new_centroids / bins_min_clamped.unsqueeze(-1)

        centroids = torch.where(zero_mask.unsqueeze(-1), centroids, new_centroids)

    return centroids, bins


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=50,
        decay=0.99,
        eps=1e-5,
        threshold_ema_dead_code=2,
    ):
        super().__init__()

        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_sum", embed.clone())

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1

    def _broadcast_buffers(self):
        if not self._is_distributed():
            return

        for buffer in self.buffers():
            dist.broadcast(buffer.data, src=0)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        if self._is_distributed():
            all_data = torch.empty(
                WORLD_SIZE * data.shape[0],
                *data.shape[1:],
                dtype=data.dtype,
                device=data.device,
            )
            dist.all_gather_into_tensor(all_data, data.contiguous())
            if IS_MAIN_PROCESS:
                embed, cluster_size = kmeans(
                    all_data, self.codebook_size, self.kmeans_iters
                )

                self.embed.data.copy_(embed)
                self.embed_sum.data.copy_(embed.clone())
                self.cluster_size.data.copy_(cluster_size)

            dist.barrier()

            self._broadcast_buffers()
        else:
            embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
            self.embed.data.copy_(embed)
            self.embed_sum.data.copy_(embed.clone())
            self.cluster_size.data.copy_(cluster_size)

        self.inited.data.copy_(torch.Tensor([True]))
        if self._is_distributed():
            dist.broadcast(self.inited.data, src=0)

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask.unsqueeze(-1), sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = einops.rearrange(batch_samples, "... d -> (...) d")

        if not self._is_distributed() or IS_MAIN_PROCESS:
            self.replace_(batch_samples, mask=expired_codes)

        if self._is_distributed():
            dist.barrier()

        if self._is_distributed():
            dist.broadcast(self.embed.data, src=0)

    def preprocess(self, x):
        """flatten all but last dimension"""
        return einops.rearrange(x, "... d -> (...) d")

    def quantize(self, x):
        # (n, d) -> (d, n)
        embed = self.embed.t()

        # compute distance between each x and embeddings
        dists = -1 * (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dists.max(dim=-1).indices

        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        """unflatten dimensions"""
        return embed_ind.reshape(*shape[:-1])

    def dequantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed)

    @torch.no_grad()
    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    @torch.no_grad()
    def decode(self, embed_ind):
        return self.dequantize(embed_ind)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype

        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)

        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)

        quantized = self.dequantize(embed_ind)

        if self.training:
            if self._is_distributed():
                batch_samples = torch.empty(
                    WORLD_SIZE * x.shape[0],
                    *x.shape[1:],
                    dtype=x.dtype,
                    device=x.device,
                )
                dist.all_gather_into_tensor(batch_samples, x.contiguous())
            else:
                batch_samples = x

            self.expire_codes_(batch_samples)

            local_cluster_count = embed_onehot.sum(dim=0)

            # (dim, n) @ (n, k) -> (dim, k)
            local_embed_sum = x.t() @ embed_onehot

            global_cluster_count = local_cluster_count
            global_embed_sum = local_embed_sum

            if self._is_distributed():
                dist.all_reduce(global_cluster_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_embed_sum, op=dist.ReduceOp.SUM)

            ema_inplace(self.cluster_size, global_cluster_count, self.decay)
            ema_inplace(self.embed_sum, global_embed_sum.t(), self.decay)

            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
                * self.cluster_size.sum()
            )

            embed_normalized = self.embed_sum / cluster_size.unsqueeze(-1)
            self.embed.data.copy_(embed_normalized)

            # Sync all buffers across GPUs to ensure consistency (could be small numerical inconsistencies)
            if self._is_distributed():
                self._broadcast_buffers()

            if self._is_distributed():
                # wait for everyone
                dist.barrier()

        return quantized, embed_ind


class VectorQuantization(nn.Module):
    """Standard VQ method"""

    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        decay=0.99,
        eps=1e-5,
        kmeans_init=True,
        kmeans_iters=50,
        threshold_ema_dead_code=2,
        commitment_weight=1,
    ):
        super().__init__()

        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.commitment_weight = commitment_weight

        codebook_dim = codebook_dim if codebook_dim is not None else dim
        requires_projection = codebook_dim != dim
        self.proj_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )

        # Setup codebook #
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

    @torch.no_grad()
    def encode(self, x):
        x = einops.rearrange(x, "b d n -> b n d")
        x = self.proj_in(x)
        embed_ind = self._codebook.encode(x)
        return embed_ind

    @torch.no_grad()
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.proj_out
        quantize = einops.rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device

        x = einops(x, "b d n -> b n d")

        x = self.proj_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.proj_out(quantize)
        quantize = einops.rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    def __init__(self, n_quantizers, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(n_quantizers)]
        )

    @torch.no_grad()
    def encode(self, x, n_q=None):
        residual = x
        all_indices = []

        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantize = layer.decode(indices)
            residual = residual - quantize
            all_indices.append(indices)

        out_indices = torch.stack(all_indices)
        return out_indices

    @torch.no_grad()
    def decode(self, q_indices):
        quantize_out = torch.tensor(0.0, device=q_indices.device)

        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantize = layer.decode(indices)
            quantize_out = quantize_out + quantize

        return quantize_out

    def forward(self, x, n_q=None):
        quantize_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            quantize, indices, losses = layer(residual)

            residual = residual - quantize.detach()
            quantize_out = quantize_out + quantize

            all_indices.append(indices)
            all_losses.append(losses)

        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices)

        return quantize_out, out_indices, out_losses


def test_codebook():
    from accelerate import Accelerator

    accelerator = Accelerator()

    # Test parameters
    dim = 128
    codebook_size = 16
    seq_len = 100
    batch_size = 8

    # Create codebook
    codebook = EuclideanCodebook(
        dim=dim,
        codebook_size=codebook_size,
        kmeans_init=True,
        threshold_ema_dead_code=10,
    )

    # Move to device and prepare
    codebook = codebook.to(accelerator.device)
    codebook.train()

    accelerator.print(f"\n{'=' * 50}")
    accelerator.print("Testing EuclideanCodebook with DDP")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Current process: {accelerator.process_index}")
    accelerator.print(f"{'=' * 50}\n")

    # Test initialization
    accelerator.print("Step 1: Testing k-means initialization...")
    x = torch.randn(batch_size, seq_len, dim).to(accelerator.device)

    # First forward pass (triggers k-means init)
    quantized, indices = codebook(x)

    accelerator.wait_for_everyone()

    # Check if embeddings are synced across GPUs
    if accelerator.is_main_process:
        embed_sum = codebook.embed.sum().item()
        cluster_size_sum = codebook.cluster_size.sum().item()
        accelerator.print("✓ Initialization complete")
        accelerator.print(f"  Embed sum: {embed_sum:.4f}")
        accelerator.print(f"  Cluster size sum: {cluster_size_sum:.4f}")

    accelerator.wait_for_everyone()

    # Test multiple training steps
    accelerator.print("\nStep 2: Testing training updates...")
    for step in range(3):
        x = torch.randn(batch_size, dim).to(accelerator.device)
        quantized, indices = codebook(x)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            embed_sum = codebook.embed.sum().item()
            cluster_size_sum = codebook.cluster_size.sum().item()
            accelerator.print(
                f"  Step {step + 1}: embed_sum={embed_sum:.4f}, cluster_size_sum={
                    cluster_size_sum:.4f}"
            )

    # Verify synchronization across GPUs
    accelerator.print("\nStep 3: Verifying buffer synchronization...")

    # Gather embeddings from all processes
    if accelerator.num_processes > 1:
        embed_gathered = accelerator.gather(codebook.embed)
        cluster_size_gathered = accelerator.gather(codebook.cluster_size)

        if accelerator.is_main_process:
            # Check if all processes have identical embeddings
            embed_split = embed_gathered.view(
                accelerator.num_processes, codebook_size, dim
            )
            cluster_size_split = cluster_size_gathered.view(
                accelerator.num_processes, codebook_size
            )

            embed_diff = (
                (embed_split[0].unsqueeze(0) - embed_split[1:]).abs().max().item()
            )
            cluster_diff = (
                (cluster_size_split[0].unsqueeze(0) - cluster_size_split[1:])
                .abs()
                .max()
                .item()
            )

            accelerator.print(f"Max embedding difference across GPUs: {embed_diff:.2e}")
            accelerator.print(
                f"Max cluster_size difference across GPUs: {cluster_diff:.2e}"
            )

            if embed_diff < 1e-5 and cluster_diff < 1e-5:
                accelerator.print("SUCCESS: All buffers are properly synchronized!")
            else:
                accelerator.print("FAILED: Buffers are not synchronized!")
    else:
        accelerator.print("Single process mode - no synchronization needed")

    accelerator.print(f"\n{'=' * 50}\n")

    accelerator.end_training()


if __name__ == "__main__":
    # test_kmeans()
    test_codebook()
