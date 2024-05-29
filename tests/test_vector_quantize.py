import torch
import pytest
from vector_quantize_pytorch import VectorQuantize

class TestVectorQuantizer:

    vq = VectorQuantize(
        dim = 256,
        codebook_size = 512,     # codebook size
        decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight = 1.   # the weight on the commitment loss
    )

    def test_init(self):
        assert self.vq

    def test_forward(self):
        x = torch.randn(1, 1024, 256)
        quantized, indices, _ = self.vq(x)

        assert x.shape == quantized.shape
        assert indices.shape == (1, 1024)