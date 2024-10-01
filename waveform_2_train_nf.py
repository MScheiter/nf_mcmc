import numpy as np
import torch

import generative_model as gm
import utils

second_step = False

nflows = 4
dropout = 0.4
lr = 1e-4
epochs = 5000 if second_step else 40000
nbatch = 5000

percentages = np.array([0.2, 0.4, 0.6, 0.8, 1.0])           # different length of chain
N = 100000

for percentage in percentages:

    if second_step:
        data = utils.pload('data/nf_pyprop8_mcmc_samples_perc' + str(perc) + '_step1' + '.obj')
    else:
        # prepare training data
        perc = int(100 * percentage)
        thinning = int(40 * percentage)                         # equal amount of training samples for each case
        n_samples = int(100000 * percentage)

        # Get McMC ensemble
        sampler = utils.pload('data/emcee_pyprop8_lvz_long_long.obj')
        data = sampler.get_chain()[:n_samples:thinning].reshape(-1, 10)

    # Train NF on it (!! takes a while !!)
    if second_step:
        model = torch.load('data/nf_pyprop8_perc' + str(perc)  + '_step1' + '.pt')
    else:
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
    if second_step:
        torch.save(model, 'data/nf_pyprop8_perc' + str(perc) + '_nextgen_step2_alt3' + '.pt')
        utils.psave(data_nf, 'data/nf_pyprop8_samples_perc' + str(perc) + '_nextgen_step2_alt3' + '.obj')
        utils.psave(log_prob, 'data/nf_pyprop8_log_prob_perc' + str(perc) + '_nextgen_step2_alt3' + '.obj')
        utils.psave(log_post, 'data/nf_pyprop8_log_post_perc' + str(perc) + '_nextgen_step2_alt3' + '.obj')
    else:
        torch.save(model, 'data/nf_pyprop8_perc' + str(perc) + '_step1' + '.pt')
        utils.psave(data_nf, 'data/nf_pyprop8_samples_perc' + str(perc) + '_step1' + '.obj')
        utils.psave(log_prob, 'data/nf_pyprop8_log_prob_perc' + str(perc) + '_step1' + '.obj')
        utils.psave(log_post, 'data/nf_pyprop8_log_post_perc' + str(perc) + '_step1' + '.obj')
