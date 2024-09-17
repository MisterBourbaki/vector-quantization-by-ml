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

    return [
        series,
        images,
        video,
    ]


@pytest.fixture
def get_vectors():
    def _get_vectors(dim: int, channel_last: bool = True):
        if channel_last:
            series = {"feature": torch.randn(1, 100, dim), "indice_shape": (1, 100)}
            images = {"feature": torch.randn(1, 8, 8, dim), "indice_shape": (1, 8, 8)}
            video = {
                "feature": torch.randn(1, 10, 8, 8, dim),
                "indice_shape": (1, 10, 8, 8),
            }
        else:
            series = {"feature": torch.randn(1, dim, 100), "indice_shape": (1, 100)}
            images = {"feature": torch.randn(1, dim, 8, 8), "indice_shape": (1, 8, 8)}
            video = {
                "feature": torch.randn(1, dim, 10, 8, 8),
                "indice_shape": (1, 10, 8, 8),
            }

        return [
            series,
            images,
            video,
        ]

    return _get_vectors
