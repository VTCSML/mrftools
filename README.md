# mrftools

Python package to do discrete Markov random field learning and inference. The key advantage of
mrftools is that the message-passing algorithms for inference are fast yet implemented in high-level
Python. This is done by casting low-level message passing indexing as sparse-matrix operations.

If you use this work in an academic study, please cite our paper
<code>
@inproceedings{bixler:uai18,
	Author = {Reid Bixler and Bert Huang},
	Booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence},
	Title = {Sparse-Matrix Belief Propagation},
	Year = {2018}}
</code>

For now, this software is very experimental, but a lot of useful parts are functioning.
It is available as open-source software under a Creative Commons Attribution License.

The embedded documentation is available on the web at http://mrftools.readthedocs.org,
and this same documentation is embedded into the code.

# Installation

Clone the repository, then in the root folder of the repository, run
<code>python setup.py install</code>. This command should install the library
into your current Python environment.

# Requirements

The library is tested in Python 3.6 and 2.7. Its main requirements are
scipy and numpy. Some of the secondary classes require PIL and matplotlib,
but these are not critical, unless you are doing image analysis.

# Examples

We are working on building full examples of usage, but for now the unit tests are
the best source of example usage of the various classes in <code>mrftools</code>.

# GPU Support

In our UAI 2018 paper, we experimented with GPU support by using PyTorch. For now, this
is only available in our <code>pytorch-test</code> branch of the repository. This will be
better organized soon, but for now, that branch includes Torch versions of the
belief propagator class. We promise this will be cleaned up soon. (We're deciding how best
to manage the PyTorch dependency; we don't want it to be a requirement for core mrftools
installations.)

# Models

## Markov Nets

The MarkovNet class stores the basic structure and factorization of a pairwise
Markov random field.

## Log Linear Models

# Inference Classes

# Learning Classes

# Limitations

The library currently only supports pairwise Markov random fields. It does not
support sequential message passing; all message passing is done in parallel.
