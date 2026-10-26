from benchmarks.benchmarks_dkl.gp_benchmarks import (
    SUPPORTED_DATASETS,
    benchmark_dkl_vs_sdrgp_boston,
    benchmark_dkl_vs_sdrgp_dataset,
    benchmark_dkl_vs_sdrgp_energy,
)
from benchmarks.benchmarks_dkl.image_classification_benchmarks import (
    SUPPORTED_IMAGE_DATASETS,
    benchmark_dkl_vs_sdrgp_image_classification,
)

__all__ = [
    "SUPPORTED_DATASETS",
    "SUPPORTED_IMAGE_DATASETS",
    "benchmark_dkl_vs_sdrgp_boston",
    "benchmark_dkl_vs_sdrgp_dataset",
    "benchmark_dkl_vs_sdrgp_energy",
    "benchmark_dkl_vs_sdrgp_image_classification",
]
