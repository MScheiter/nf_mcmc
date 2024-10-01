import numpy as np

import emcee

import examples
import utils

pdfs = examples.LayeredModel(fixed_z=True, lvz=True)

nchains = 20
target = 100000

sampler = emcee.EnsembleSampler(nchains, pdfs.ndim, pdfs.log_post,
    moves=[(emcee.moves.WalkMove(), 0.5), (emcee.moves.StretchMove(), 0.5)])
init = np.random.rand(nchains, pdfs.ndim)
sampler.run_mcmc(init, target, progress=True)   # !! this takes several days !!
data = sampler.get_chain()

utils.psave(sampler, '../chains_pp/emcee_pyprop8_lvz_long_long.obj')
