from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor, cdist
from torch.nn import Identity

from vector_quantize_pytorch.utils.general import batched_sample_vectors, noop
from vector_quantize_pytorch.utils.losses import l2norm


def batched_bincount(batch_labels: Tensor, minlength: int) -> Tensor:
    """Returns the count of labels in the input, batchwise.

    Parameters
    ----------
    batch_labels : Tensor
        a non 1D tensor
    minlength : int
        minimal number of bins

    Returns
    -------
    Tensor
        the batched count of bins from the input tensor.
    """
    batch, dtype, device = (
        batch_labels.shape[0],
        batch_labels.dtype,
        batch_labels.device,
    )
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(batch_labels)
    target.scatter_add_(-1, batch_labels, values)
    return target


def kmeans(
    vectors: Tensor,
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim=False,
    sample_fn: Callable = batched_sample_vectors,
    all_reduce_fn: Callable = noop,
) -> tuple[Tensor, Tensor]:
    """Performs Kmeans algorithm on the provided vectors with 'num_clusters' clusters.

    The initialization of the centroids is done using the 'sample_fn' parameter.
    The algorithm can be used with euclidean similarities, with 'use_cosine_sim' set to False,
    and with cosine similarity in the case 'use_cosine_sim' set to True. In that case, the centroids are L2-normalized.

    The function returns both centroids tensor and the number of elements in each class/cluster.

    Parameters
    ----------
    vectors : Tensor
        the data on which to perform the Kmeans algorithm
    num_clusters : int
        the number of clusters to compute
    num_iters : int, optional
        the number of iteration for the algorithm, by default 10
    use_cosine_sim : bool, optional
        whether or not to use the cosine similarity distance in place of the euclidean one, by default False
    sample_fn : Callable, optional
        the "sampling" function, that is how to initialize the centroids, by default batched_sample_vectors
    all_reduce_fn : Callable, optional
        a reduction function, useful when doing distributed computations, by default noop

    Returns
    -------
    tuple[Tensor, Tensor]
        first the centroids, second the number of elements per class/clusters.
    """
    num_codebooks, dim, dtype = (
        vectors.shape[0],
        vectors.shape[-1],
        vectors.dtype,
    )

    centroids = sample_fn(vectors, num_clusters)
    if use_cosine_sim:
        # similarity_fn = CosineSimilarity(dim=-1)
        def similarity_fn(x, y):
            return x @ rearrange(y, "B N D -> B D N")

        reg_fn = l2norm
    else:

        def similarity_fn(x, y):
            return -cdist(x, y)

        reg_fn = Identity()

    for _ in range(num_iters):
        similarities = similarity_fn(vectors, centroids)

        class_labels = torch.argmax(similarities, dim=-1)
        num_per_class = batched_bincount(class_labels, minlength=num_clusters)
        all_reduce_fn(num_per_class)

        zero_mask = num_per_class == 0
        bins_min_clamped = num_per_class.masked_fill(zero_mask, 1)

        new_centroids = class_labels.new_zeros(
            num_codebooks, num_clusters, dim, dtype=dtype
        )

        new_centroids.scatter_add_(
            1, repeat(class_labels, "h n -> h n d", d=dim), vectors
        )
        new_centroids = new_centroids / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_centroids)

        new_centroids = reg_fn(new_centroids)

        centroids = torch.where(
            rearrange(zero_mask, "... -> ... 1"), centroids, new_centroids
        )

    return centroids, num_per_class
