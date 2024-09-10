# Changelog

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
