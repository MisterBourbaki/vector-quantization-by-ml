from rich.traceback import install

from vector_quantization.finite_scalar_quantization import FSQ
from vector_quantization.latent_quantization import LatentQuantize
from vector_quantization.lookup_free_quantization import LFQ
from vector_quantization.random_projection_quantizer import (
    RandomProjectionQuantizer,
)
from vector_quantization.residual_fsq import GroupedResidualFSQ, ResidualFSQ
from vector_quantization.residual_lfq import GroupedResidualLFQ, ResidualLFQ
from vector_quantization.residual_vq import GroupedResidualVQ, ResidualVQ
from vector_quantization.vector_quantize_pytorch import VectorQuantize

install(show_locals=False)

__all__ = [
    "FSQ",
    "LatentQuantize",
    "LFQ",
    "RandomProjectionQuantizer",
    "GroupedResidualFSQ",
    "GroupedResidualLFQ",
    "GroupedResidualVQ",
    "ResidualFSQ",
    "ResidualLFQ",
    "ResidualVQ",
    "VectorQuantize",
]
