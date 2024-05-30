# Changelog

## Work on vector_quantize_pytorch module

### Added

* Add a `codebooks` module, holding both `EuclideanCodebook` and `CosineSimCodebook`, and their utilities.
* Add a `utils` module.
* Add a dedicated test suite in `tests/test_vector_quantize.py`.
* The gumbel_sample parameter is now typed with Callable.
* Pydantic dependency, may be useful for configurations.
* Marimo, wily and rope to help the refacto. Might remove them at the end.
* BaseCodebook class as a base class for codebooks.

### Changed

* the `transform_input` parameter is now defined through lambda function.
* the buffer `initt` is replaced by a simple attribute `is_kmeans_init`.
* The initialization of the Kmeans is simplified by moving the check of whether it has been initialized out of the method `init_embed_`.

### Removed

* Functions `Sequential` and `exists`, which were not necessary.
* Function `identity` function, replaced in EuclideanCodebook by a simple lambda function.
* Function `l2norm`, simply replaced by normalize.