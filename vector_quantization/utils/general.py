from math import ceil

import torch
import torch.nn.functional as F
from einops import pack, repeat, unpack
from torch import Tensor, nn


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def identity(t):
    return t


def sample_vectors(vector: Tensor, num_of_samples: int) -> Tensor:
    """Sample randomly num_of_samples vectors from the vector.

    The input tensor is of shape (N, *). The function runs as follow:
        * if N is greater than num_of_samples, then the output is a random collection of num_of_samples vector.
        * if N is less than num_of_samples, then the output is a collection of possibly repeated, randomly selected vectors from the input.

    In both cases, the output tensor is of shape (num_of_samples, *).

    Parameters
    ----------
    vector : Tensor
        a tensor of shape (N, *)
    num_of_samples : int
        the number of samples to take from the vector

    Returns
    -------
    Tensor
        a tensor of shape (num_of_samples, *)
    """
    num_of_vec, device = vector.shape[0], vector.device
    if num_of_vec >= num_of_samples:
        indices = torch.randperm(num_of_vec, device=device)[:num_of_samples]
    else:
        indices = torch.randint(0, num_of_vec, (num_of_samples,), device=device)

    return vector[indices]


def batched_sample_vectors(batch: Tensor, num_of_samples: int) -> Tensor:
    """Return a batch of sampled vectors from the input batch.

    Parameters
    ----------
    batch : Tensor
        a tensor (batch of vectors) of shape (B, N, *)
    num_of_samples : int
        the number of samples to take from the each vector of the batch

    Returns
    -------
    Tensor
        a batch of vectors of shape (B, num_of_samples, *)
    """
    return torch.stack(
        [sample_vectors(vector, num_of_samples) for vector in batch.unbind(dim=0)],
        dim=0,
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


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    reinmax=False,
    dim=-1,
    training=True,
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (
        reinmax and not straight_through
    ), "reinmax can only be turned on if using straight through gumbel softmax"

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        prob0 = logits.softmax(dim=dim)
        prob1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        prob1 = ((log(prob1) - logits).detach() + logits).softmax(dim=1)
        prob2 = 2 * prob1 - 0.5 * prob0
        one_hot = prob2 - prob2.detach() + one_hot
    else:
        prob1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + prob1 - prob1.detach()

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)
