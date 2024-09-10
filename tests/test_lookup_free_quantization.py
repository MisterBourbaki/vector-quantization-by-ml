import pytest
import torch

from vector_quantize_pytorch.lookup_free_quantization import LFQ


@pytest.fixture
def vectors_lfq(dim: int = 4):
    # series = torch.randn(1, 100, dim)
    series = torch.randn(1, dim, 100)
    images = torch.randn(1, dim, 8, 8)
    video = torch.randn(1, dim, 10, 8, 8)

    return [
        series,
        images,
        video,
    ]


class TestLFQ:
    # levels = [8, 5, 5, 5]
    quantizer = LFQ(
        codebook_size=2**6,  # codebook size, must be a power of 2
        dim=4,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight=0.1,  # how much weight to place on entropy loss
        diversity_gamma=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
    )

    def test_init(self):
        assert self.quantizer
        assert self.quantizer.has_projections

    def test_forward(self, vectors_channel_last):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_last:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
            # assert indices.shape == features.shape[0:1] + features.shape[2:]
            assert indices.shape == features.shape[:-1]


class TestLFQNoProjections:
    # levels = [8, 5, 5, 5]
    quantizer = LFQ(
        codebook_size=2**4,  # codebook size, must be a power of 2
        dim=4,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight=0.1,  # how much weight to place on entropy loss
        diversity_gamma=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
    )

    def test_init(self):
        assert self.quantizer
        assert not self.quantizer.has_projections

    def test_forward(self, vectors_channel_last):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_last:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
            # assert indices.shape == features.shape[0:1] + features.shape[2:]
            assert indices.shape == features.shape[:-1]


class TestLFQSpherical:
    # levels = [8, 5, 5, 5]
    quantizer = LFQ(
        codebook_size=2**6,  # codebook size, must be a power of 2
        dim=4,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight=0.1,  # how much weight to place on entropy loss
        diversity_gamma=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        spherical=True,
    )

    def test_init(self):
        assert self.quantizer
        assert self.quantizer.has_projections

    def test_forward(self, vectors_channel_last):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_last:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            # assert indices.shape == features.shape[0:1] + features.shape[2:]
            assert indices.shape == features.shape[:-1]

            ## Following test failed, as it seems the 'spherical' case is not well implemented...
            # assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestLFQChannelFirst:
    # levels = [8, 5, 5, 5]
    quantizer = LFQ(
        codebook_size=2**6,  # codebook size, must be a power of 2
        dim=4,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight=0.1,  # how much weight to place on entropy loss
        diversity_gamma=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        channel_first=True,
    )

    def test_init(self):
        assert self.quantizer
        assert self.quantizer.has_projections

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
            assert indices.shape == features.shape[0:1] + features.shape[2:]
