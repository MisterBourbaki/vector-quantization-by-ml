from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import distributed, nn
from torch.optim import Optimizer

from vector_quantize_pytorch.codebooks import CosineSimCodebook, EuclideanCodebook
from vector_quantize_pytorch.utils import (
    default,
    gumbel_sample,
    orthogonal_loss_fn,
    pack_one,
    unpack_one,
)


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=False,
        decay=0.8,
        eps=1e-5,
        freeze_codebook=False,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        use_cosine_sim=False,
        layernorm_after_project_in=False,  # proposed by @SaltyChtao here https://github.com/lucidrains/vector-quantize-pytorch/issues/26#issuecomment-1324711561
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=1.0,
        commitment_use_cross_entropy_loss=False,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        stochastic_sample_codes=False,
        sample_codebook_temp=1.0,
        straight_through=False,
        reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook=None,
        sync_affine_param=False,
        ema_update=True,
        learnable_codebook=False,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
        sync_update_v=0.0,  # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim

        if requires_projection and layernorm_after_project_in:
            self.project_in = nn.Sequential(
                nn.Linear(dim, codebook_input_dim),
                nn.LayerNorm(codebook_input_dim),
            )
        elif requires_projection:
            self.project_in = nn.Linear(dim, codebook_input_dim)
        else:
            self.project_in = nn.Identity()

        self.project_out = (
            nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        )

        self.has_projections = requires_projection

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss  # whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (
            ema_update and learnable_codebook
        ), "learnable codebook not compatible with EMA update"

        assert 0 <= sync_update_v <= 1.0
        assert not (
            sync_update_v > 0.0 and not learnable_codebook
        ), "learnable codebook must be turned on"

        self.sync_update_v = sync_update_v

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic=stochastic_sample_codes,
            reinmax=reinmax,
            straight_through=straight_through,
        )

        if sync_codebook is None:
            sync_codebook = (
                distributed.is_initialized() and distributed.get_world_size() > 1
            )

        codebook_kwargs = dict(
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sync_kmeans=sync_kmeans,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp=sample_codebook_temp,
            gumbel_sample=gumbel_sample_fn,
            ema_update=ema_update,
        )

        if affine_param:
            assert (
                not use_cosine_sim
            ), "affine param is only compatible with euclidean codebook"
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param=True,
                sync_affine_param=sync_affine_param,
                affine_param_batch_decay=affine_param_batch_decay,
                affine_param_codebook_decay=affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = (
            in_place_codebook_optimizer(self._codebook.parameters())
            if in_place_codebook_optimizer
            else None
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, "1 ... -> ...")

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, "... -> 1 ...")

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
        else:
            indices, ps = pack_one(indices, "b * h")
            indices = rearrange(indices, "b n h -> b h n")

            indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
            codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

            codes = codebook.gather(2, indices)
            codes = rearrange(codes, "b h n d -> b n (h d)")
            codes = unpack_one(codes, ps, "b * d")

        if not self.channel_last:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def forward(
        self,
        x,
        indices=None,
        mask=None,
        sample_codebook_temp=None,
        freeze_codebook=False,
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert mask is None
            x = rearrange(x, "b d -> b 1 d")

        shape, device, heads, is_multiheaded, codebook_size, return_loss = (
            x.shape,
            x.device,
            self.heads,
            self.heads > 1,
            self.codebook_size,
            indices is not None,
        )

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = self.in_place_codebook_optimizer is not None

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c")

        if need_transpose:
            x = rearrange(x, "b d n -> b n d")

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp=sample_codebook_temp,
            mask=mask,
            freeze_codebook=freeze_codebook,
        )

        # quantize

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:
            if mask:
                loss = F.mse_loss(quantize, x.detach(), reduction="none")

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(
                        mask,
                        "b n -> c (b h) n",
                        c=loss.shape[0],
                        h=loss.shape[1] // mask.shape[0],
                    )

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # quantize again

            quantize, embed_ind, distances = self._codebook(
                x, **codebook_forward_kwargs
            )

        if self.training:
            # determine code to use for commitment loss
            if not self.learnable_codebook or freeze_codebook:
                commit_quantize = quantize.detach()
            else:
                commit_quantize = quantize

            # straight through

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.0:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (
                    quantize - quantize.detach()
                )

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = "1 b n l -> b l n"
            elif self.separate_codebook_per_head:
                dist_einops_eq = "c b n l -> b l n c"
            else:
                dist_einops_eq = "1 (b h) n l -> b l n h"

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b=shape[0]), codes, ignore_index=-1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(
                embed_ind, "b (h w) ... -> b h w ...", h=height, w=width
            )

        if only_one:
            embed_ind = rearrange(embed_ind, "b 1 ... -> b ...")

        # aggregate loss

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if mask:
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, "b n -> b n h", h=heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                elif mask:
                    # with variable lengthed sequences
                    commit_loss = F.mse_loss(commit_quantize, x, reduction="none")

                    loss_mask = mask
                    if is_multiheaded:
                        loss_mask = repeat(
                            loss_mask,
                            "b n -> c (b h) n",
                            c=commit_loss.shape[0],
                            h=commit_loss.shape[1] // mask.shape[0],
                        )

                    commit_loss = commit_loss[loss_mask].mean()
                else:
                    commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (
                        is_multiheaded and self.separate_codebook_per_head
                    ), "orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet"
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if (
                    self.orthogonal_reg_max_codes
                    and num_codes > self.orthogonal_reg_max_codes
                ):
                    rand_ids = torch.randperm(num_codes, device=device)[
                        : self.orthogonal_reg_max_codes
                    ]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, "b n d -> b d n")

        if self.accept_image_fmap:
            quantize = rearrange(quantize, "b (h w) c -> b c h w", h=height, w=width)

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")

        # if masking, only return quantized for where mask has True

        if mask:
            quantize = torch.where(
                rearrange(mask, "... -> ... 1"), quantize, orig_input
            )

        return quantize, embed_ind, loss
