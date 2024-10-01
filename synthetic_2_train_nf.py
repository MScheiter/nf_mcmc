import numpy as np
import torch
import tqdm

import generative_model as gm
import utils

nflows = 4
nhid = 256
nbatch = 5000
lr = 1e-4
epochs = 100
dropout = 0.4

N = 10000

ndims = 2, 4, 6, 10, 15, 20, 25

for ndim in ndims:

    # prepare training data
    sampler = utils.pload('github/data/sampler_circle_' + str(ndim) + '_1.obj')
    data = sampler.get_chain(flat=True, thin=20, discard=2000)

    # Train NF on it
    model = gm.RealNVP(sampler.ndim, nflows, dropout=dropout)
    model.optimize(lr, epochs, nbatch, data)

    model.eval()

    # generate samples
    data_nf = model.sample(N=N)

    # compute NF's density for the whole batch
    z, prob = model.f(torch.Tensor(data_nf).to(model.device))
    log_prob = model.prior.log_prob(z) + prob
    log_prob = log_prob.cpu().detach().numpy()

    # calculate true posterior values of NF-proposed samples (!! loop takes a long time !!)
    log_post = np.zeros(N)
    for i in tqdm.tqdm(range(N)):
        log_post[i] = sampler.log_prob_fn(data_nf[i])

    # save all of this
    torch.save(model, 'data/nf_circle_ndim_' + str(ndim) + '.pt')
    utils.psave(data_nf, 'data/nf_circle_samples_ndim_' + str(ndim) + '.obj')
    utils.psave(log_prob, 'data/nf_circle_log_prob_ndim_' + str(ndim) + '.obj')
    utils.psave(log_post, 'data/nf_circle_log_post_ndim_' + str(ndim) + '.obj')
