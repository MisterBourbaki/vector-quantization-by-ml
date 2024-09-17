# Changelog

## [Unreleased]

* Create a new, meta Codebook class which can be used to create both EuclideanCodebook and CosineSimCodebook. Allows for further types of Codebooks.
* Replace the "distance" function by similarity function, defined through the 'use_cosine_sim' boolean. May be replaced by an Enum later on.
* Add weights regularization and transform input as booleans in the parameters of the Codebook.
* Use fully the parameter codebook_params in VQ to define the Codebook, Cosine or not.
* Transform gumbel_params and kmeans_params into dict if they are still dataclasses at initialization. This avoid error if the kmeans_params and gumbel_params come from 'asdict'-ification of CodebookParams early on.
* CHange the tests accordingly.
* Rename the method 'replace' in Codebook into 'replace_codes' to avoid confusion with 'replace' from Dataclasses library.


## [Version 1.16.3]

* Introduce dataclass GumbelParams for the sampling function (here, various kinds of Gumbel sampling).
* Remove corresponding old parameters/attributes, gathered now into the new dataclass.
* Remove the creation of the sampling function as a method in VQ, into the codebook itself.
* Remove the definition of sample_codebook_temp inside the forward method: there is no reason now to do so.
* Remove sample_codebook_temp as a parameter of the forward method in both VQ and Codebook classes.
* Change accordingly the ResidualVQ class.

## [Version 1.16.2]

* Introduce dataclasses KmeansParams and AffineParams to gather parameters useful for only one feature.
* Change VQ and codeooks accordingly. This allow for better readibility (dataclasses can be documented) and reduce the huge number of parameters for VQ and codebooks classes.
* Introduce dataclasses for Codebooks Parameters. Use in VQ as a proxy for now, as the definition/declaration of both dim/codebook_dim and codebook_size is a bit more intricate.

## [Version 1.16.1]

* Replace the ad-hoc 'cdist' function by the built-in torch.cdist function.
* In kmeans, move the 'if/else use_cos_sim' a layer above in terms of logic. At start, define both 'dist_fn' and 'ref_fn' depending on whether or not the user wants to use cosine similarity. This change allows for more flexibility for other and future computation of distances. Also the definition is not repeated at each iteration, gaining obvious speed gain.
* Documentation of 'sample_vectors' and 'batched_sample_vectors' and rewriting of the name of variables for clarification.
* In 'kmeans', rename of several variables for clarification.
* Add tests for initialization through kmeans, with or without cosine similarity, and in the case of multihead.
* Documentation of the kmeans function.
* In codebooks, renaming some variables and attributes regarding the use of kmeans initialization, for clarity.
* Documentation and clarification of 'batched_bincount' function.
* Change the level at which the check on whether the embeddings have initialized or not, that is now before calling the method initialize_embeddings and not inside it. Improve performance and clarity.
* Adapt change of name for attribute embed to embeddings in ResidualVQ.

## [Version 1.16.0]

* [BUG FIXED] In lookup_freee_quantization: fix the bug of 'should_transpose', badly defined. The forward method should transpose regardless of 'is or is not image', just regarding the 'channel_first' attribute. If 'is_image_or_video', the input tensor should be packed (regardless of channel first or not.)
* [BUG FIXED] the VQ API could not handle video, and was not the same as the other ones. The definition of 'need_transpose' was not appropriate, as it depended in an hidden way on whether the input tensors was or not an image. The code now looks more stable, as it handles many more situation: image or not, channel last or not, multi headed or not.
* Remove: the attribute 'accept_img_map', unnecessary.
* CHange the test suite for lookup free quantization according to the previous point. Now the API is common to other VQ classes (except still the base one...).
* Remove the 'default' function, really unnecessary.
* [BUG FIXED] dealing with multi headed codebooks, regardless of whether it is image or not, channel last or not.
* Remove the test class for VQImage, as the API is now fixed for all kind of tensors.
* Improve code suite cases using the pytest fixtures, defining attributes like dim, codebook_dim and so on.
* Remove the 'accept_img_map' attribute from ResidualVQ too.
* [BUG FIXED] int 'GroupedResidualVQ', the 'split_dim' property attribute used 'accept_image_fmap', where it is only linked to the 'channel_last' attribute.

* Add a test suite for VectorQuantizer.
* Add 'vectors' fixtures in a dedicated conftest.py to provide some common data input, with or without channel first.
* Remove the 'test_readme.py' file, as it is definitely not useful. Will be replaced in the future by the pytest-examples plugin.
