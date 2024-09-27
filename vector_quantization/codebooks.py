from dataclasses import asdict, dataclass, field, is_dataclass
from functools import partial

import torch
from einops import rearrange, reduce, repeat
from torch import cdist, distributed, einsum, nn
from torch.amp import autocast
from torch.nn import Module

from vector_quantization.utils.distributed import (
    maybe_distributed_mean,
    sample_vectors_distributed,
)
from vector_quantization.utils.general import (
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
from vector_quantization.utils.kmeans import kmeans
from vector_quantization.utils.losses import l2norm


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


@dataclass
class GumbelParams:
    temperature: float = 1.0
    stochastic: bool = False
    reinmax: bool = False
    straight_through: bool = False
    dim: int = -1
    training: bool = True


@dataclass
class CodebookParams:
    dim: int
    codebook_size: int
    num_codebooks: int = 1
    initialization_by_kmeans: bool = False
    kmeans_params: KmeansParameters = None
    decay: float = 0.8
    eps_for_smoothing: float = 1e-5
    threshold_ema_dead_code: int = 2
    reset_cluster_size: int = None
    use_ddp: bool = False
    distributed_replace_codes: bool = True
    learnable_codebook: bool = False
    gumbel_params: GumbelParams = field(default_factory=GumbelParams)
    ema_update: bool = True
    use_affine: bool = False
    affine_params: AffineParameters = None
    transform_input: str = "identity"
    use_cosine_sim: bool = False
    weights_regularization: str = "identity"


class Codebook(Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        initialization_by_kmeans: bool = False,
        kmeans_params: KmeansParameters = None,
        decay: float = 0.8,
        eps_for_smoothing: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        reset_cluster_size: int = None,
        use_ddp: bool = False,
        distributed_replace_codes: bool = True,
        learnable_codebook: bool = False,
        gumbel_params: GumbelParams = GumbelParams(),
        ema_update: bool = True,
        use_affine: bool = False,
        affine_params: AffineParameters = None,
        transform_input: str = "identity",
        use_cosine_sim: bool = False,
        weights_regularization: str = "identity",
    ):
        super().__init__()
        if transform_input == "identity":
            self.transform_input = identity
        elif transform_input == "l2norm":
            self.transform_input = l2norm
        else:
            raise f"The option {transform_input} as transform input function is not yet implemented"

        if weights_regularization == "identity":
            self.weights_regularization = identity
        elif weights_regularization == "l2norm":
            self.weights_regularization = l2norm
        else:
            raise f"The option {weights_regularization} for weights regularization is not yet implemented."

        self.use_cosine_sim = use_cosine_sim
        if use_cosine_sim:

            def similarity_fn(x, y):
                return einsum("h n d, h c d -> h n c", x, y)

            self.similarity_fn = similarity_fn
        else:

            def similarity_fn(x, y):
                return -cdist(x, y)

            self.similarity_fn = similarity_fn

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not initialization_by_kmeans else torch.zeros
        embeddings = self.weights_regularization(
            init_fn(num_codebooks, codebook_size, dim)
        )

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_params = (
            asdict(kmeans_params) if is_dataclass(kmeans_params) else kmeans_params
        )
        self.eps_for_smoothing = eps_for_smoothing
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = (
            reset_cluster_size
            if reset_cluster_size is not None
            else threshold_ema_dead_code
        )
        if is_dataclass(gumbel_params):
            gumbel_params = asdict(gumbel_params)
        self.sample_fn_training = partial(gumbel_sample, **gumbel_params)
        gumbel_params["training"] = False
        self.sample_fn_val = partial(gumbel_sample, **gumbel_params)

        assert not (
            use_ddp and num_codebooks > 1 and initialization_by_kmeans
        ), "kmeans init is not compatible with multiple codebooks in distributed environment for now"

        self.sample_fn = (
            sample_vectors_distributed
            if use_ddp and self.kmeans_params["sync"]
            else batched_sample_vectors
        )

        self.distributed_replace_codes = distributed_replace_codes
        self.replace_sample_fn = (
            sample_vectors_distributed
            if use_ddp and self.kmeans_params["sync"] and distributed_replace_codes
            else batched_sample_vectors
        )

        self.kmeans_all_reduce_fn = (
            distributed.all_reduce if use_ddp and self.kmeans_params["sync"] else noop
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

        # affine related params

        self.use_affine = use_affine
        if use_affine:
            self.affine_params = affine_params

            self.register_buffer("batch_mean", None)
            self.register_buffer("batch_variance", None)

            self.register_buffer("codebook_mean_needs_init", torch.Tensor([True]))
            self.register_buffer("codebook_mean", torch.empty(num_codebooks, 1, dim))
            self.register_buffer("codebook_variance_needs_init", torch.Tensor([True]))
            self.register_buffer(
                "codebook_variance", torch.empty(num_codebooks, 1, dim)
            )

    @torch.jit.ignore
    def initialize_embeddings(self, data, mask=None):
        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        # note that if use_cosine_sim is True, then the embeddings are already "regularized"
        embeddings, cluster_size = kmeans(
            data,
            num_clusters=self.codebook_size,
            num_iters=self.kmeans_params["iter"],
            use_cosine_sim=self.use_cosine_sim,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embeddings * rearrange(cluster_size, "... -> ... 1")

        self.embeddings.data.copy_(embeddings)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)

    def replace_codes(self, batch_samples, batch_mask):
        batch_samples = self.weights_regularization(batch_samples)
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
        self.replace_codes(batch_samples, batch_mask=expired_codes)

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

    @autocast(device_type="cuda", enabled=False)
    def forward(self, x, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4

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

        similarities = self.similarity_fn(flatten, embeddings)

        embed_ind, embed_onehot = self.sample_fn_training(
            similarities,
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
            embed_normalized = self.weights_regularization(embed_normalized)
            self.embeddings.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(
                lambda t: rearrange(t, "1 ... -> ..."), (quantize, embed_ind)
            )

        similarities = unpack_one(similarities, ps, "h * d")

        return quantize, embed_ind, similarities
