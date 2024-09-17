import torch

from vector_quantization.finite_scalar_quantization import FSQ


class TestFSQ:
    levels = [8, 5, 5, 5]
    quantizer = FSQ(levels)

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_last:
            quantized, indices = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestFSQNoIndices:
    levels = [8, 5, 5, 5]
    quantizer = FSQ(levels, return_indices=False)

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        for features in vectors_channel_last:
            features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
            _, indices = self.quantizer(features)

            assert indices is None


class TestFSQWithDimAndChannelFirst:
    levels = [8, 5]
    dim = 4
    quantizer = FSQ(levels, dim=dim, channel_first=True)

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestFSQSeveralCodebooks:
    levels = [8, 5]
    quantizer = FSQ(levels, num_codebooks=2)

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_last:
            quantized, indices = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))


class TestFSQSeveralCodebooksKeepCodebooks:
    levels = [8, 5]
    num_codebooks = 2
    quantizer = FSQ(levels, num_codebooks=num_codebooks, keep_num_codebooks_dim=True)

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_last):
        for features in vectors_channel_last:
            quantized, indices = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
            assert indices.shape[-1] == self.num_codebooks


class TestFSQWithDimAndChannelFirstSeveralCodebooks:
    levels = [8, 5]
    dim = 4
    quantizer = FSQ(levels, dim=dim, channel_first=True, num_codebooks=2)

    def test_init(self):
        assert self.quantizer

    def test_forward(self, vectors_channel_first):
        # features = torch.randn(1, 1024, 4)  # 4 since there are 4 levels
        for features in vectors_channel_first:
            quantized, indices = self.quantizer(features)

            assert quantized.shape == features.shape
            assert torch.all(quantized == self.quantizer.indices_to_codes(indices))
