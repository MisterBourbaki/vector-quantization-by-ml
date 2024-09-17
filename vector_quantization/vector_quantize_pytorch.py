from collections import namedtuple
from dataclasses import asdict, replace
from typing import Callable

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer

from vector_quantization.codebooks import (
    Codebook,
    CodebookParams,
)
from vector_quantization.utils.distributed import (
    is_distributed,
)
from vector_quantization.utils.general import (
    entropy,
    exists,
    identity,
    unpack_one,
)
from vector_quantization.utils.losses import orthogonal_loss_fn

LossBreakdown = namedtuple(
    "LossBreakdown",
    [
        "commitment",
        "codebook_diversity",
        "orthogonal_reg",
        "inplace_optimize",
    ],
)


class VectorQuantize(Module):
    def __init__(
        self,
        dim,
        codebook_params: CodebookParams,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=False,
        layernorm_after_project_in=False,  # proposed by @SaltyChtao here https://github.com/lucidrains/vector-quantize-pytorch/issues/26#issuecomment-1324711561
        channel_last=True,
        commitment_weight=1.0,
        commitment_use_cross_entropy_loss=False,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        codebook_diversity_loss_weight=0.0,
        codebook_diversity_temperature=100.0,
        sync_codebook=None,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # Optimizer used if using learnable_codebook
        sync_update_v=0.0,  # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = codebook_dim if codebook_dim is not None else dim
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

        self.has_commitment_loss = commitment_weight > 0.0
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss  # whether to use cross entropy loss to codebook as commitment loss

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0.0

        if not exists(sync_codebook):
            sync_codebook = is_distributed()

        self.codebook_params = replace(
            codebook_params,
            dim=codebook_dim,
            num_codebooks=heads if separate_codebook_per_head else 1,
            learnable_codebook=has_codebook_orthogonal_loss
            or codebook_params.learnable_codebook,
            use_ddp=sync_codebook,
        )

        self.learnable_codebook = codebook_params.learnable_codebook

        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        has_codebook_diversity_loss = codebook_diversity_loss_weight > 0.0
        self.has_codebook_diversity_loss = has_codebook_diversity_loss
        self.codebook_diversity_temperature = codebook_diversity_temperature
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight

        assert not (
            codebook_params.ema_update and codebook_params.learnable_codebook
        ), "learnable codebook not compatible with EMA update"

        assert 0 <= sync_update_v <= 1.0
        assert not (
            sync_update_v > 0.0 and not codebook_params.learnable_codebook
        ), "learnable codebook must be turned on"

        self.sync_update_v = sync_update_v

        self._codebook = Codebook(**asdict(self.codebook_params))

        self.in_place_codebook_optimizer = (
            in_place_codebook_optimizer(self._codebook.parameters())
            if exists(in_place_codebook_optimizer)
            else None
        )

        self.channel_last = channel_last

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

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
            indices, ps = pack([indices], "b * h")
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
        freeze_codebook=False,
        return_loss_breakdown=False,
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not mask is not None
            x = rearrange(x, "b d -> b 1 d")

        shape, device, heads, is_multiheaded, return_loss = (
            x.shape,
            x.device,
            self.heads,
            self.heads > 1,
            exists(indices),
        )

        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        is_img_or_video = x.ndim >= 4

        if not self.channel_last:
            x = rearrange(x, "b d ... -> b ... d")
        if is_img_or_video:
            x, ps = pack([x], "b * d")

        x = self.project_in(x)

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        x = self._codebook.transform_input(x)
        codebook_forward_kwargs = dict(
            mask=mask,
            freeze_codebook=freeze_codebook,
        )

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        commit_loss = orthogonal_reg_loss = inplace_optimize_loss = (
            codebook_diversity_loss
        ) = self.zero

        if should_inplace_optimize and self.training and not freeze_codebook:
            if mask is not None:
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

            inplace_optimize_loss = loss

            quantize, embed_ind, distances = self._codebook(
                x, **codebook_forward_kwargs
            )

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = (
                torch.detach
                if not self.learnable_codebook or freeze_codebook
                else identity
            )

            commit_quantize = maybe_detach(quantize)

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

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        if is_img_or_video and not is_multiheaded:
            embed_ind = unpack_one(embed_ind, ps, "b *")
        elif is_img_or_video and is_multiheaded:
            embed_ind = unpack_one(embed_ind, ps, "b * h")

        if only_one:
            embed_ind = rearrange(embed_ind, "b 1 ... -> b ...")

        # aggregate loss

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            # calculate codebook diversity loss (negative of entropy) if needed

            if self.has_codebook_diversity_loss:
                prob = (-distances * self.codebook_diversity_temperature).softmax(
                    dim=-1
                )
                avg_prob = reduce(prob, "... n l -> n l", "mean")
                codebook_diversity_loss = -entropy(avg_prob).mean()

                loss = (
                    loss + codebook_diversity_loss * self.codebook_diversity_loss_weight
                )

            # commitment loss

            if self.has_commitment_loss:
                if self.commitment_use_cross_entropy_loss:
                    if mask is not None:
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, "b n -> b n h", h=heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                elif mask is not None:
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
                    exists(self.orthogonal_reg_max_codes)
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

        if is_img_or_video:
            quantize = unpack_one(quantize, ps, "b * d")
        if not self.channel_last:
            quantize = rearrange(quantize, "b ... d -> b d ...")
        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")

        # if masking, only return quantized for where mask has True

        if mask is not None:
            quantize = torch.where(
                rearrange(mask, "... -> ... 1"), quantize, orig_input
            )

        if not return_loss_breakdown:
            return quantize, embed_ind, loss

        loss_breakdown = LossBreakdown(
            commit_loss,
            codebook_diversity_loss,
            orthogonal_reg_loss,
            inplace_optimize_loss,
        )

        return quantize, embed_ind, loss, loss_breakdown
