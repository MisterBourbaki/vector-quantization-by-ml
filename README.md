# Vector Quantization Using Machine/Deep Learning algorithms

## Why this project?

This project is a fork from the vector-quantize-pytorch package, the go-to library for vector quantization using ML methods in Pytorch. This fork aims at building a new, simple API for all kind of VQ algorithms, from the classic one to more recent ones as LFQ of Latent Quantization.

The original code lacks readability, full test suite, documentation, and so on, and I hope to provide some remedies to those in the near future.

## Vector Quantization

Vector Quantization is a classical quantization technique that allows the representation of vectors with "dense" values (typically, vectors in [0,1]^d) as vectors or even number with finite range. This is particularly useful (and is the origin of such techniques) in case where compression of the data is important.

The topic gained traction these past few years as it came to researchers that introducing a VQ algorithm inside latent space of a very large model implies sometimes an improvement in the task at hand. This is due to the ability of such algorithms to retain only the very useful features of, for instance, latent vectors.

## Work in Progress

This project is definitely not even in Beta, as a lot needs to be done in terms of refactoring the whole codebase. The main goal is to fuse all different APIs from FSQ, LFQ, VQ and so on into one common, simple API. All VQ-like algorithms should also be compatible with any kind of input, which is not the case at all for the oldest of algorithm in the original codebase.

Another goal is to improve performances where it can be done, and where it can be useful.
