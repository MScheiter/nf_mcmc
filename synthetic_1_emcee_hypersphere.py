import numpy as np

import emcee

import examples
import utils

ndims = 2, 4, 6, 10, 15, 20, 25

pdfs = examples.Circle(ndim=ndim)

for ndim in ndims:

    nchains = 50
    target = 10000

    samplers = []
    samplers.append(emcee.EnsembleSampler(nchains, ndim,pdfs.log_post, moves=emcee.moves.GaussianMove(0.5)))
    samplers.append(emcee.EnsembleSampler(nchains, ndim,pdfs.log_post))

    for j, sampler in enumerate(samplers):

        init = np.random.rand(nchains, ndim)
        sampler.run_mcmc(init, target, progress=True)

        utils.psave(sampler, 'data/sampler_circle_' + str(ndim) + '_' + str(j) + '.obj')
