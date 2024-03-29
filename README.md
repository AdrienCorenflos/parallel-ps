# Parallel particle smoothing

This is the companion code to the paper "De-Sequentialized Monte Carlo: a parallel-in-time particle smoother", by Adrien
Corenflos, Nicolas Chopin, and Simo Särkkä. Pre-print is available on ArXiv at https://arxiv.org/abs/2202.02264.
The full paper is published at JMLR https://www.jmlr.org/papers/v23/22-0140.html. Please cite this version.

## Quick description

This package implements parallel-in-time smoothing methods for state-space methods. By this we mean that the runtime of
the algorithm on parallel hardware (such as GPU) will be proportional to $log(T)$ where $T$ is the number of required
time steps.

The way we achieve this is by re-phrasing the smoothing problem as a divide-and-conquer operation over partial
smoothing. In order to do inference in this now nested structure, we require that one is able to sample from proposals
independent marginals `q_t` at each time t. This can be done for example by computing a parallel-in-time approximate
LGSSM smoother.

We moreover implement a parallel-in-time particle Gibbs sampler. Because our sampled smoothing trajectories suffer
degeneracy uniformly across time, we do not need a backward sampling pass to prevent degeneracy in time for efficient
rejuvenation.

Finally, we implement a lazy resampling scheme to improve the sampling capabilities of our algorithm.

All the examples are located in the `examples` folder.  

For more details, we refer to our article.

## Installation

This package has different requirements depending on your intended use for it.

### Minimal installation

If you simply want to use it as part of your own code, then we recommend the following steps

1. Create and activate a virtual environment using your favourite method (`conda create ...` or `python -m venv path`
   for example).
2. Install your required version of JAX:
    * GPU (preferred): at the time of writing JAX **only supports the GPU backend for linux distributions**. You will
      need to make sure you have the proper CUDA (at the time of writing 11.5) version installed and then run (at the
      time of writing)
      ```bash
      pip install --upgrade pip
      pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html --ignore-installed # Note: wheels only available on linux.
      ```
    * CPU (no support for parallelisation): at the time of writing this is supported for **linux and MacOS** users only.
      This should already be taken care of by the `chex` package requirement, but, if not, run (at the time of writing)
    ```bash
     pip install --upgrade pip
     pip install --upgrade "jax[cpu]"
     ```
    * For more up-to-date installation instructions we refer to JAX github page https://github.com/google/jax.
3. Run `pip install -r requirements.txt`
4. Run `python setup.py [develop|install]` depending on if you plan to work on the source code or not.

### Additional test requirements

If you plan on running the tests, please run `pip install -r requirements-tests.txt`

### Additional examples requirements

If you plan on running the examples located in the examples folder, please
run `pip install -r requirements-examples.txt`

## Contact information

This library was developed by Adrien Corenflos. For any code related question feel free to open a discussion in the
issues tab, and for more technical questions please email the article corresponding author 
adrien[dot]corenflos[at]aalto[dot]fi.

## How to cite

If you like and use our code/build on our work, please cite us!
