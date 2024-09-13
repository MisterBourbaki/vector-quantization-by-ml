from dataclasses import dataclass
from functools import partial

import torch
from einops import rearrange, reduce, repeat
from torch import cdist, distributed, einsum, nn
from torch.cuda.amp import autocast
from torch.nn import Module

from vector_quantize_pytorch.utils.distributed import (
    maybe_distributed_mean,
    sample_vectors_distributed,
)
from vector_quantize_pytorch.utils.general import (
    batched_embedding,
    batched_sample_vectors,
    ema_inplace,
    exists,
    gumbel_sample,
    identity,
    laplace_smoothing,
    noop,
    pack_one,
    uniform_init,
    unpack_one,
)
from vector_quantize_pytorch.utils.kmeans import kmeans
from vector_quantize_pytorch.utils.losses import l2norm


@dataclass
class AffineParameters:
    """Dataclass gathering parameters for affine update."""

    sync: bool
    batch_decay: float = 0.99
    codebook_decay: float = 0.9


@dataclass
class KmeansParameters:
    """Dataclass gathering parameters for Kmeans algorithm."""

    iter: int = 10
    sync: bool = True


class EuclideanCodebook(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        initialization_by_kmeans: bool = False,
        kmeans_params: KmeansParameters = None,
        decay=0.8,
        eps_for_smoothing=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        distributed_replace_codes=True,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        use_affine=False,
        affine_params: AffineParameters = None,
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not initialization_by_kmeans else torch.zeros
        embeddings = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_params = kmeans_params
        self.eps_for_smoothing = eps_for_smoothing
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = (
            reset_cluster_size
            if reset_cluster_size is not None
            else threshold_ema_dead_code
        )

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (
            use_ddp and num_codebooks > 1 and initialization_by_kmeans
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = (
            sample_vectors_distributed
            if use_ddp and self.kmeans_params.sync
            else batched_sample_vectors
        )

        self.distributed_replace_codes = distributed_replace_codes
        self.replace_sample_fn = (
            sample_vectors_distributed
            if use_ddp and self.kmeans_params.sync and distributed_replace_codes
            else batched_sample_vectors
        )

        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and self.kmeans_params.sync else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        # self.register_buffer("is_initialized", torch.Tensor([not initialization_by_kmeans]))
        self.is_initialized = not initialization_by_kmeans
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embeddings.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer("embeddings", embeddings)

        # affine related params

        self.use_affine = use_affine
        self.affine_params = affine_params

        self.register_buffer("batch_mean", None)
        self.register_buffer("batch_variance", None)

        self.register_buffer("codebook_mean_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_mean", torch.empty(num_codebooks, 1, dim))
        self.register_buffer("codebook_variance_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_variance", torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def initialize_embeddings(self, data, mask=None):
        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embeddings, cluster_size = kmeans(
            data,
            num_clusters=self.codebook_size,
            num_iters=self.kmeans_params.iter,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embeddings * rearrange(cluster_size, "... -> ... 1")

        self.embeddings.data.copy_(embeddings)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        # self.is_initialized.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not exists(old_value) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embeddings, mask=None):
        assert self.use_affine

        var_fn = partial(torch.var, unbiased=False)

        # calculate codebook mean and variance

        embeddings = rearrange(embeddings, "h ... d -> h (...) d")

        if self.training:
            self.update_with_decay(
                "codebook_mean",
                reduce(embeddings, "h n d -> h 1 d", "mean"),
                self.affine_params.codebook_decay,
            )
            self.update_with_decay(
                "codebook_variance",
                reduce(embeddings, "h n d -> h 1 d", var_fn),
                self.affine_params.codebook_decay,
            )

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, "h ... d -> h (...) d")

        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        # calculate batch mean and variance

        if not self.affine_params.sync:
            self.update_with_decay(
                "batch_mean",
                reduce(data, "h n d -> h 1 d", "mean"),
                self.affine_params.batch_decay,
            )
            self.update_with_decay(
                "batch_variance",
                reduce(data, "h n d -> h 1 d", var_fn),
                self.affine_params.batch_decay,
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

        self.update_with_decay(
            "batch_mean",
            batch_mean,
            self.affine_params.batch_decay,
        )

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, "h n d -> h 1 d", "sum")
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay(
            "batch_variance",
            batch_variance,
            self.affine_params.batch_decay,
        )

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(
                rearrange(samples, "... -> 1 ..."), mask.sum().item()
            )
            sampled = rearrange(sampled, "1 ... -> ...")

            if not self.distributed_replace_codes:
                sampled = maybe_distributed_mean(sampled)

            self.embeddings.data[ind][mask] = sampled
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
        sample_codebook_temp = (
            sample_codebook_temp
            if sample_codebook_temp is not None
            else self.sample_codebook_temp
        )
        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        flatten, ps = pack_one(x, "h * d")

        if mask is not None:
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )
        if not self.is_initialized:
            self.initialize_embeddings(flatten, mask=mask)
            self.is_initialized = True

        if self.use_affine:
            self.update_affine(flatten, self.embeddings, mask=mask)

        embeddings = (
            self.embeddings if self.learnable_codebook else self.embeddings.detach()
        )

        if self.use_affine:
            codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
            embeddings = (embeddings - self.codebook_mean) * (
                batch_std / codebook_std
            ) + self.batch_mean

        dist = -cdist(flatten, embeddings)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embeddings)
        else:
            quantize = batched_embedding(embed_ind, embeddings)

        if self.training and self.ema_update and not freeze_codebook:
            if self.use_affine:
                flatten = (flatten - self.batch_mean) * (
                    codebook_std / batch_std
                ) + self.codebook_mean

            if mask is not None:
                embed_onehot[~mask] = 0.0

            cluster_size = embed_onehot.sum(dim=1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps_for_smoothing
            ) * self.cluster_size.sum(dim=-1, keepdim=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
            self.embeddings.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(
                lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind)
            )

        dist = unpack_one(dist, ps, "h * d")

        return quantize, embed_ind, dist


class CosineSimCodebook(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        initialization_by_kmeans=False,
        kmeans_params: KmeansParameters = None,
        decay=0.8,
        eps_for_smoothing=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        distributed_replace_codes=True,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay

        if not initialization_by_kmeans:
            embeddings = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embeddings = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_params = kmeans_params
        self.eps_for_smoothing = eps_for_smoothing
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = (
            reset_cluster_size
            if reset_cluster_size is not None
            else threshold_ema_dead_code
        )

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = (
            sample_vectors_distributed
            if use_ddp and self.kmeans_params.sync
            else batched_sample_vectors
        )

        self.distributed_replace_codes = distributed_replace_codes
        self.replace_sample_fn = (
            sample_vectors_distributed
            if use_ddp and self.kmeans_params.sync and distributed_replace_codes
            else batched_sample_vectors
        )

        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and self.kmeans_params.sync else noop
        )
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.is_initialized = not initialization_by_kmeans
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embeddings.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer("embeddings", embeddings)

    @torch.jit.ignore
    def initialize_embeddings(self, data, mask=None):
        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embeddings, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_params.iter,
            use_cosine_sim=True,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embeddings * rearrange(cluster_size, "... -> ... 1")

        self.embeddings.data.copy_(embeddings)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples, batch_mask)):
            sampled = self.replace_sample_fn(
                rearrange(samples, "... -> 1 ..."), mask.sum().item()
            )
            sampled = rearrange(sampled, "1 ... -> ...")

            if not self.distributed_replace_codes:
                sampled = maybe_distributed_mean(sampled)

            self.embeddings.data[ind][mask] = sampled
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
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = (
            sample_codebook_temp
            if sample_codebook_temp is not None
            else self.sample_codebook_temp
        )

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        flatten, ps = pack_one(x, "h * d")

        if mask is not None:
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )

        if not self.is_initialized:
            self.initialize_embeddings(flatten, mask=mask)
            self.is_initialized = True

        embeddings = (
            self.embeddings if self.learnable_codebook else self.embeddings.detach()
        )

        dist = einsum("h n d, h c d -> h n c", flatten, embeddings)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )
        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embeddings)
        else:
            quantize = batched_embedding(embed_ind, embeddings)

        if self.training and self.ema_update and not freeze_codebook:
            if mask is not None:
                embed_onehot[~mask] = 0.0

            bins = embed_onehot.sum(dim=1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size.data, bins, self.decay)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps_for_smoothing
            ) * self.cluster_size.sum(dim=-1, keepdim=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
            embed_normalized = l2norm(embed_normalized)

            self.embeddings.data.copy_(l2norm(embed_normalized))
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(
                lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind)
            )

        dist = unpack_one(dist, ps, "h * d")
        return quantize, embed_ind, dist
