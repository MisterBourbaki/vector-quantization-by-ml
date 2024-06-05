from functools import partial
from typing import Callable, Optional

import torch
from einops import rearrange, reduce, repeat
from torch import distributed, einsum, nn
from torch.cuda.amp import autocast
from torch.nn.functional import normalize

from vector_quantize_pytorch.utils import (
    default,
    gumbel_sample,
    pack_one,
    unpack_one,
)


def noop(*args, **kwargs):
    pass


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (
        (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy)
        .clamp(min=0)
        .sqrt()
    )


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack(
        [sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0
    )


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(
            num, all_num_samples / all_num_samples.sum()
        )
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


## TODO:
# Use this https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html#sphx-glr-auto-tutorials-kmeans-plot-kmeans-torch-py
# to improve the kmeans implementation
def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    use_cosine_sim=False,
    sample_fn=batched_sample_vectors,
    all_reduce_fn=noop,
):
    num_codebooks, dim, dtype, device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, "h n d -> h d n")
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = normalize(new_means, p=2, dim=-1)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        learnable_codebook=False,
        gumbel_sample: Callable = gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        affine_param=False,
        sync_affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
    ):
        super().__init__()
        self.transform_input = lambda x: x

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
            use_ddp and num_codebooks > 1 and kmeans_init
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = (
            sample_vectors_distributed
            if use_ddp and sync_kmeans
            else batched_sample_vectors
        )
        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and sync_kmeans else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.is_kmeans_init = not kmeans_init
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer("batch_mean", None)
        self.register_buffer("batch_variance", None)

        self.register_buffer("codebook_mean_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_mean", torch.empty(num_codebooks, 1, dim))
        self.register_buffer("codebook_variance_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_variance", torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if mask:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * rearrange(cluster_size, "... -> ... 1")

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.is_kmeans_init = True

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if old_value is not None or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask=None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased=False)

        # calculate codebook mean and variance

        embed = rearrange(embed, "h ... d -> h (...) d")

        if self.training:
            self.update_with_decay(
                "codebook_mean",
                reduce(embed, "h n d -> h 1 d", "mean"),
                self.affine_param_codebook_decay,
            )
            self.update_with_decay(
                "codebook_variance",
                reduce(embed, "h n d -> h 1 d", var_fn),
                self.affine_param_codebook_decay,
            )

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, "h ... d -> h (...) d")

        if mask:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.update_with_decay(
                "batch_mean",
                reduce(data, "h n d -> h 1 d", "mean"),
                self.affine_param_batch_decay,
            )
            self.update_with_decay(
                "batch_variance",
                reduce(data, "h n d -> h 1 d", var_fn),
                self.affine_param_batch_decay,
            )
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator

        num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)
        distributed.all_reduce(num_vectors)

        # calculate distributed mean

        batch_sum = reduce(data, "h n d -> h 1 d", "sum")
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay("batch_mean", batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, "h n d -> h 1 d", "sum")
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay(
            "batch_variance", batch_variance, self.affine_param_batch_decay
        )

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(
            zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))
        ):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(
                rearrange(samples, "... -> 1 ..."), mask.sum().item()
            )
            sampled = rearrange(sampled, "1 ... -> ...")

            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        dtype = x.dtype
        flatten, ps = pack_one(x, "h * d")

        if mask:
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )
        if not self.is_kmeans_init:
            self.init_embed_(flatten, mask=mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (
                batch_std / codebook_std
            ) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (
                    codebook_std / batch_std
                ) + self.codebook_mean

            if mask:
                embed_onehot[~mask] = 0.0

            cluster_size = embed_onehot.sum(dim=1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps
            ) * self.cluster_size.sum(dim=-1, keepdim=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize = rearrange(quantize, "1 ... -> ...")
            embed_ind = rearrange(embed_ind, "1 ... -> ...")

        dist = unpack_one(dist, ps, "h * d")

        return quantize, embed_ind, dist


class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        learnable_codebook=False,
        gumbel_sample: Callable = gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
    ):
        super().__init__()
        self.transform_input = lambda x: normalize(x, p=2, dim=-1)

        self.ema_update = ema_update
        self.decay = decay

        if not kmeans_init:
            embed = normalize(
                uniform_init(num_codebooks, codebook_size, dim), p=2, dim=-1
            )
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = (
            sample_vectors_distributed
            if use_ddp and sync_kmeans
            else batched_sample_vectors
        )
        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and sync_kmeans else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.is_kmeans_init = not kmeans_init
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if mask:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * rearrange(cluster_size, "... -> ... 1")

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.is_kmeans_init = True

    def replace(self, batch_samples, batch_mask):
        batch_samples = normalize(batch_samples, p=2, dim=-1)

        for ind, (samples, mask) in enumerate(
            zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0))
        ):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(
                rearrange(samples, "... -> 1 ..."), mask.sum().item()
            )
            sampled = rearrange(sampled, "1 ... -> ...")

            self.embed.data[ind][mask] = sampled
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size
            self.cluster_size.data[ind][mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(
        self,
        x,
        sample_codebook_temp: Optional[float] = None,
        mask=None,
        freeze_codebook=False,
    ):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = (
            self.sample_codebook_temp
            if sample_codebook_temp is None
            else sample_codebook_temp
        )

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        flatten, ps = pack_one(x, "h * d")

        if mask:
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )
        if not self.is_kmeans_init:
            self.init_embed_(flatten, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        dist = einsum("h n d, h c d -> h n c", flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )
        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            self.update_codebooks(x, mask, flatten, embed_onehot)

        if needs_codebook_dim:
            quantize = rearrange(quantize, "1 ... -> ...")
            embed_ind = rearrange(embed_ind, "1 ... -> ...")

        dist = unpack_one(dist, ps, "h * d")
        return quantize, embed_ind, dist

    def update_codebooks(self, x, mask, flatten, embed_onehot):
        if mask:
            embed_onehot[~mask] = 0.0

        bins = embed_onehot.sum(dim=1)
        self.all_reduce_fn(bins)

        ema_inplace(self.cluster_size.data, bins, self.decay)

        embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
        embed_sum = embed_sum.contiguous()
        self.all_reduce_fn(embed_sum)

        ema_inplace(self.embed_avg.data, embed_sum, self.decay)

        cluster_size = laplace_smoothing(
            self.cluster_size, self.codebook_size, self.eps
        ) * self.cluster_size.sum(dim=-1, keepdim=True)

        embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
        embed_normalized = normalize(embed_normalized, p=2, dim=-1)

        self.embed.data.copy_(normalize(embed_normalized, p=2, dim=-1))
        self.expire_codes_(x)
