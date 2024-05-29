import torch
import torch.nn.functional as F
from einops import pack, unpack
from torch import einsum, nn


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def identity(t):
    return t


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


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
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return None
    elif len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)
