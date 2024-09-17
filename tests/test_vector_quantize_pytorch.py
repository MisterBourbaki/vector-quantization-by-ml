import torch

from vector_quantization import VectorQuantize
from vector_quantization.codebooks import CodebookParams, KmeansParameters


class TestVectorQuantizer:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)

    quantizer = VectorQuantize(
        dim=dim,
        commitment_weight=1.0,  # the weight on the commitment loss
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_not_use_cosine(self):
        assert not self.quantizer._codebook.use_cosine_sim

    def test_forward(self, vectors_channel_last):
        vectors = vectors_channel_last

        for features in vectors:
            quantized, indices, _ = self.quantizer(features)

            assert quantized.shape == features.shape
            assert indices.shape == features.shape[:-1]


class TestVectorQuantizerChannelFirst:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)

    quantizer = VectorQuantize(
        dim=dim,
        commitment_weight=1.0,  # the weight on the commitment loss
        channel_last=False,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim, channel_last=False)

        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert quantized.shape == features["feature"].shape
            assert indices.shape == features["indice_shape"]


class TestVectorQuantizerCosine:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(
        dim=dim, codebook_size=codebook_size, use_cosine_sim=True
    )
    quantizer = VectorQuantize(
        dim=4,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_use_cosine(self):
        assert self.quantizer._codebook.use_cosine_sim

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim)
        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert features["feature"].shape == quantized.shape
            assert indices.shape == features["indice_shape"]


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
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim)

        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert features["feature"].shape == quantized.shape
            assert indices.shape == features["indice_shape"] + (self.heads,)


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
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim)

        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert features["feature"].shape == quantized.shape
            assert indices.shape == features["indice_shape"] + (self.heads,)


class TestVectorQuantizerLowerCode:
    dim = 4
    codebook_size = 2**5
    codebook_params = CodebookParams(dim=dim, codebook_size=codebook_size)
    quantizer = VectorQuantize(
        dim=4,
        codebook_dim=2,  # paper proposes setting this to 32 or as low as 8 to increase codebook usage
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim)
        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert features["feature"].shape == quantized.shape
            assert indices.shape == features["indice_shape"]


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
        commitment_weight=1.0,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim)
        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert features["feature"].shape == quantized.shape
            assert indices.shape == features["indice_shape"]


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
        use_cosine_sim=True,
    )

    quantizer = VectorQuantize(
        dim=4,
        commitment_weight=1.0,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self, get_vectors):
        vectors = get_vectors(dim=self.dim)
        for features in vectors:
            quantized, indices, _ = self.quantizer(features["feature"])

            assert features["feature"].shape == quantized.shape
            assert indices.shape == features["indice_shape"]


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
        commitment_weight=1.0,
        codebook_params=codebook_params,
    )

    def test_init(self):
        assert self.quantizer

    def test_forward(self):
        series = torch.randn(1, self.codebook_size // 2, self.dim)
        quantized, indices, _ = self.quantizer(series)

        assert series.shape == quantized.shape
        assert indices.shape == series.shape[:-1]
