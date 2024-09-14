import torch

from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.codebooks import CodebookParams, KmeansParameters


class TestVectorQuantizer:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)

    quantizer = VectorQuantize(
        dim=dim,
        codebook_size=codebook_size,  # codebook size
        # decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight=1.0,  # the weight on the commitment loss
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        vectors = vectors_channel_last

        for features in vectors:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            # assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
            assert indices.shape == features.shape[:-1]


class TestVectorQuantizerChannelFirst:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)

    quantizer = VectorQuantize(
        dim=dim,
        codebook_size=codebook_size,  # codebook size
        # decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight=1.0,  # the weight on the commitment loss
        channel_last=False,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        vectors = vectors_channel_first

        for features in vectors:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            # assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
            assert indices.shape == features.shape[0:1] + features.shape[2:]


class TestVectorQuantizerCosine:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)
    quantizer = VectorQuantize(
        # dim=256,
        dim=4,
        codebook_size=2**5,  # codebook size
        use_cosine_sim=True,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        # series = torch.randn(1, 1024, 256)
        series = torch.randn(1, 1024, 4)
        quantized, indices, _ = self.quantizer(series)

        assert series.shape == quantized.shape
        assert indices.shape == (1, 1024)


class TestVectorQuantizerMultihead:
    codebook_dim = 32
    heads = 2
    dim = codebook_dim * heads
    codebook_size = 2**5

    codebook_params = CodebookParams(dim=codebook_dim, codebook_size=codebook_size)
    quantizer = VectorQuantize(
        dim=dim,
        codebook_dim=codebook_dim,  # a number of papers have shown smaller codebook dimension to be acceptable
        heads=heads,  # number of heads to vector quantize, codebook shared across all heads
        separate_codebook_per_head=True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        codebook_size=2**5,
        # accept_image_fmap=True,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        # TODO: how to change the 'dim' parameter of vectors_channel_last?
        images = torch.randn(1, 8, 8, self.dim)
        series = torch.randn(1, 100, self.dim)
        video = torch.randn(1, 5, 8, 8, self.dim)
        vectors = [images, series, video]

        for features in vectors:
            quantized, indices, _ = self.quantizer(features)

            assert features.shape == quantized.shape
            assert indices.shape == features.shape[:-1] + (self.heads,)

    # def test_forward(self, vectors_channel_last):
    #     vectors = vectors_channel_last(self.dim)

    #     for features in vectors:
    #         quantized, indices, _ = self.quantizer(features)

    #         assert quantized.shape == features.shape
    #         # assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
    #         assert indices.shape == features.shape[:-1] + (self.heads,)


class TestVectorQuantizerMultiheadWithKmeansInit:
    codebook_dim = 32
    heads = 2
    dim = codebook_dim * heads
    kmeans_params = KmeansParameters()
    codebook_params = CodebookParams(
        dim=codebook_dim,
        codebook_size=2**5,
        initialization_by_kmeans=True,
        kmeans_params=kmeans_params,
    )
    quantizer = VectorQuantize(
        dim=dim,
        codebook_dim=codebook_dim,  # a number of papers have shown smaller codebook dimension to be acceptable
        heads=heads,  # number of heads to vector quantize, codebook shared across all heads
        separate_codebook_per_head=True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
        codebook_size=2**5,
        # initialization_by_kmeans=True,
        codebook_params=codebook_params,
        # accept_image_fmap=True,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        # TODO: how to change the 'dim' parameter of vectors_channel_last?
        images = torch.randn(1, 8, 8, self.dim)
        series = torch.randn(1, 100, self.dim)
        video = torch.randn(1, 5, 8, 8, self.dim)
        vectors = [images, series, video]

        for features in vectors:
            quantized, indices, _ = self.quantizer(features)

            assert features.shape == quantized.shape
            assert indices.shape == features.shape[:-1] + (self.heads,)

    # def test_forward(self, vectors_channel_last):
    #     vectors = vectors_channel_last(self.dim)

    #     for features in vectors:
    #         quantized, indices, _ = self.quantizer(features)

    #         assert quantized.shape == features.shape
    #         # assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
    #         assert indices.shape == features.shape[:-1] + (self.heads,)


class TestVectorQuantizerLowerCode:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)
    quantizer = VectorQuantize(
        dim=4,
        codebook_size=256,
        codebook_dim=2,  # paper proposes setting this to 32 or as low as 8 to increase codebook usage
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        series = torch.randn(1, 1024, 4)
        quantized, indices, _ = self.quantizer(series)

        assert series.shape == quantized.shape
        assert indices.shape == (1, 1024)


class TestVectorQuantizerKmeansInit:
    dim = 4
    initialization_by_kmeans = True
    codebook_size = 2**5
    kmeans_params = KmeansParameters()
    codebook_params = CodebookParams(
        dim=dim,
        codebook_size=codebook_size,
        initialization_by_kmeans=True,
        kmeans_params=KmeansParameters(),
    )

    quantizer = VectorQuantize(
        dim=4,
        codebook_size=codebook_size,
        # decay=0.8,
        commitment_weight=1.0,
        codebook_params=codebook_params,
        # initialization_by_kmeans=initialization_by_kmeans,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        series = torch.randn(1, self.codebook_size * 2, self.dim)
        images = torch.randn(1, 8, 8, self.dim)
        vectors = [series, images]
        for feature in vectors:
            quantized, indices, _ = self.quantizer(feature)

            assert feature.shape == quantized.shape
            assert indices.shape == feature.shape[:-1]


class TestVectorQuantizerKmeansInitWithCosine:
    dim = 4
    initialization_by_kmeans = True
    codebook_size = 2**5
    kmeans_params = KmeansParameters()
    codebook_params = CodebookParams(
        dim=dim,
        codebook_size=2**5,
        initialization_by_kmeans=True,
        kmeans_params=kmeans_params,
    )

    quantizer = VectorQuantize(
        dim=4,
        codebook_size=codebook_size,
        # decay=0.8,
        commitment_weight=1.0,
        # initialization_by_kmeans=initialization_by_kmeans,
        codebook_params=codebook_params,
        use_cosine_sim=True,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        series = torch.randn(1, self.codebook_size * 2, self.dim)
        # images = torch.randn(1, 8, 8, self.dim)
        vectors = [series]
        for feature in vectors:
            quantized, indices, _ = self.quantizer(feature)

            assert feature.shape == quantized.shape
            assert indices.shape == feature.shape[:-1]


class TestVectorQuantizerKmeansInitWithFewSamples:
    dim = 4
    initialization_by_kmeans = True
    codebook_size = 2**5
    kmeans_params = KmeansParameters()
    codebook_params = CodebookParams(
        dim=dim,
        codebook_size=2**5,
        initialization_by_kmeans=True,
        kmeans_params=kmeans_params,
    )

    quantizer = VectorQuantize(
        dim=4,
        codebook_size=codebook_size,
        # decay=0.8,
        commitment_weight=1.0,
        # initialization_by_kmeans=initialization_by_kmeans,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        series = torch.randn(1, self.codebook_size // 2, self.dim)
        quantized, indices, _ = self.quantizer(series)

        assert series.shape == quantized.shape
        assert indices.shape == series.shape[:-1]
