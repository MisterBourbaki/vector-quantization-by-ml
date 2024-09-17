import pytest
import torch

from vector_quantization.latent_quantization import LatentQuantize


class TestLatentQuantizer:
    quantizer = LatentQuantize(
        levels=[5, 5, 8],  # number of levels per codebook dimension
        # dim=16,  # input dim
        dim=4,
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestLatentQuantizerNoOptim:
    quantizer = LatentQuantize(
        levels=[5, 5, 8],  # number of levels per codebook dimension
        dim=4,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
        optimize_values=False,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestLatentQuantizerSameLevel:
    quantizer = LatentQuantize(
        levels=[5, 5, 5],  # number of levels per codebook dimension
        dim=4,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
    )

    def test_init_same_level(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestLatentQuantizerInt:
    quantizer = LatentQuantize(
        levels=5,  # number of levels per codebook dimension
        dim=4,  # input dim
        commitment_loss_weight=0.1,
        quantization_loss_weight=0.1,
        codebook_dim=3,
    )

    def test_init_int(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestLatentQuantizerBadInt:
    with pytest.raises(RuntimeError):
        quantizer_int = LatentQuantize(
            levels=5,  # number of levels per codebook dimension
            dim=16,  # input dim
            commitment_loss_weight=0.1,
            quantization_loss_weight=0.1,
            # codebook_dim=3,
        )
