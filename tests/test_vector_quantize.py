import torch

from vector_quantize_pytorch import VectorQuantize


class TestVectorQuantizer:
    vq = VectorQuantize(
        dim=256,
        codebook_size=512,  # codebook size
        decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight=1.0,  # the weight on the commitment loss
    )

    def test_init(self):
        assert self.vq

    def test_forward(self):
        x = torch.randn(1, 1024, 256)
        quantized, indices, _ = self.vq(x)

        assert x.shape == quantized.shape
        assert indices.shape == (1, 1024)


class TestVectorQuantizerCosine:
    vq = VectorQuantize(
        dim=256,
        codebook_size=512,  # codebook size
        use_cosine_sim=True,
    )

    def test_init(self):
        assert self.vq

    def test_forward(self):
        x = torch.randn(1, 1024, 256)
        quantized, indices, _ = self.vq(x)

        assert x.shape == quantized.shape
        assert indices.shape == (1, 1024)


class TestVectorQuantizerMultihead:
    vq = VectorQuantize(
        dim=256,
        codebook_dim=32,  # a number of papers have shown smaller codebook dimension to be acceptable
        heads=8,  # number of heads to vector quantize, codebook shared across all heads
        separate_codebook_per_head=True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        codebook_size=8196,
        accept_image_fmap=True,
    )

    def test_init(self):
        assert self.vq

    def test_forward(self):
        img_fmap = torch.randn(1, 256, 32, 32)
        quantized, indices, _ = self.vq(img_fmap)

        assert img_fmap.shape == quantized.shape
        assert indices.shape == (1, 32, 32, 8)


class TestVectorQuantizerLowerCode:
    vq = VectorQuantize(
        dim=256,
        codebook_size=256,
        codebook_dim=16,  # paper proposes setting this to 32 or as low as 8 to increase codebook usage
    )

    def test_init(self):
        assert self.vq

    def test_forward(self):
        x = torch.randn(1, 1024, 256)
        quantized, indices, _ = self.vq(x)

        assert x.shape == quantized.shape
        assert indices.shape == (1, 1024)
class TestVectorQuantizerNoInit:
    vq = VectorQuantize(
        dim=256,
        codebook_size=512,
        decay=0.8,  
        commitment_weight=1.0,  
        kmeans_init=True,
    )

    def test_init(self):
        assert self.vq

    def test_forward(self):
        x = torch.randn(1, 1024, 256)
        quantized, indices, _ = self.vq(x)

        assert x.shape == quantized.shape
        assert indices.shape == (1, 1024)