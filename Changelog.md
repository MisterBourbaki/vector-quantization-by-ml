# Changelog

## Work on vector_quantize_pytorch module

### Added

* Add a `codebooks` module, holding both `EuclideanCodebook` and `CosineSimCodebook`, and their utilities.
* Add a `utils` module.
* Add a dedicated test suite in `tests/test_vector_quantize.py`.
* Pydantic dependency, may be useful for configurations.
* Marimo, wily and rope to help the refacto. Might remove them at the end.

### Removed

* Functions `Sequential` and `exists`, which were not necessary.
* Function `identity` function, replaced in EuclideanCodebook by a simple lambda function.
* Function `l2norm`, simply replace by normalize.