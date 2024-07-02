import pytest
import torch


@pytest.fixture
def vectors_channel_first(dim: int = 4):
    series = torch.randn(1, dim, 100)
    images = torch.randn(1, dim, 8, 8)
    video = torch.randn(1, dim, 10, 8, 8)

    return [series, images, video]


@pytest.fixture
def vectors_channel_last(dim: int = 4):
    series = torch.randn(1, 100, dim)
    images = torch.randn(1, 8, 8, dim)
    video = torch.randn(1, 10, 8, 8, dim)

    return [series, images, video]
