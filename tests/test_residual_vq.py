import torch

from vector_quantize_pytorch.codebooks import (
    CodebookParams,
    GumbelParams,
    KmeansParameters,
)
from vector_quantize_pytorch.residual_vq import GroupedResidualVQ, ResidualVQ


class TestResidualVQ:
    dim = 4
    num_quantizers = 3
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)

    quantizer = ResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,  # specify number of quantizers
        codebook_size=2**5,  # codebook size
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        x = torch.randn(1, 100, self.dim)

        quantized, indices, commit_loss = self.quantizer(x)
        quantized, indices, commit_loss, all_codes = self.quantizer(
            x, return_all_codes=True
        )

        assert x.shape == quantized.shape
        assert indices.shape == x.shape[:-1] + (self.num_quantizers,)


class TestResidualVQ2:
    dim = 4
    num_quantizers = 3
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)
    gumbel_params = GumbelParams(stochastic=True)

    quantizer = ResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=2**5,
        gumbel_params=gumbel_params,
        # stochastic_sample_codes=True,
        # sample_codebook_temp=0.1,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        shared_codebook=True,  # whether to share the codebooks for all quantizers or not
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        x = torch.randn(1, 100, self.dim)

        quantized, indices, commit_loss = self.quantizer(x)
        quantized, indices, commit_loss, all_codes = self.quantizer(
            x, return_all_codes=True
        )

        assert x.shape == quantized.shape
        assert indices.shape == x.shape[:-1] + (self.num_quantizers,)


class TestResidualVQ3:
    dim = 4
    num_quantizers = 3
    codebook_size = 2**5
    kmeans_params = KmeansParameters()
    codebook_params = CodebookParams(
        dim=dim,
        codebook_size=codebook_size,
        initialization_by_kmeans=True,
        kmeans_params=kmeans_params,
    )

    quantizer = ResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=2**5,
        codebook_params=codebook_params,
        # initialization_by_kmeans=True,  # set to True
        # kmeans_iters=10,  # number of kmeans iterations to calculate the centroids for the codebook on init
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        x = torch.randn(1, 100, self.dim)

        quantized, indices, commit_loss = self.quantizer(x)
        quantized, indices, commit_loss, all_codes = self.quantizer(
            x, return_all_codes=True
        )

        assert x.shape == quantized.shape
        assert indices.shape == x.shape[:-1] + (self.num_quantizers,)


class TestGroupedResidualVQ:
    dim = 4
    num_quantizers = 3
    groups = 2
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)

    quantizer = GroupedResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,  # specify number of quantizers
        groups=groups,
        codebook_size=2**5,  # codebook size
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        x = torch.randn(1, 100, self.dim)

        quantized, indices, commit_loss = self.quantizer(x)
        quantized, indices, commit_loss, all_codes = self.quantizer(
            x, return_all_codes=True
        )

        assert x.shape == quantized.shape
        assert indices.shape == (self.groups,) + x.shape[:-1] + (self.num_quantizers,)


def test_residual_vq3():
    dim = 4
    # num_quantizers = 3
    codebook_size = 2**5
    kmeans_params = KmeansParameters()
    codebook_params = CodebookParams(
        dim=dim,
        codebook_size=codebook_size,
        initialization_by_kmeans=True,
        kmeans_params=kmeans_params,
    )

    quantizer = ResidualVQ(
        dim=4,
        codebook_size=2**5,
        num_quantizers=4,
        codebook_params=codebook_params,
        # initialization_by_kmeans=True,  # set to True
        # kmeans_iters=10,  # number of kmeans iterations to calculate the centroids for the codebook on init
    )

    x = torch.randn(1, 1024, 4)
    quantized, indices, commit_loss = quantizer(x)
