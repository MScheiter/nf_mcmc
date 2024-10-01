import numpy as np
import scipy.stats as stats

import emcee

import examples
import utils

pdfs = examples.Circle()

ndim = 2

m_min,m_max = -1, 1
d0,sigma_d = 0.7, 0.1
target = 7500

widths = np.logspace(-2, -0.5, 15)
ensembles = np.zeros((len(widths), target, ndim))
acc_rates = np.zeros(len(widths))
autocorrs = np.zeros(len(widths))

for i, width in enumerate(widths):
    ensemble = np.zeros((target, ndim))

    # initial model
    m = np.random.uniform(m_min, m_max, size=ndim)

    # prob of initial model
    log_post_m = pdfs.log_post(m)

    proposed = 0
    accepted = 0
    while accepted < target:

        # propose new model
        dm = np.random.normal(0, width, size=ndim)
        m_cand = m + dm
        proposed += 1

        # calculate prob of new model
        log_post_m_cand = pdfs.log_post(m_cand)

        # toss a coin
        log_alpha = np.log(np.random.uniform())

        # accept or reject
        if log_alpha < log_post_m_cand - log_post_m:
            ensemble[accepted] = m
            m = m_cand
            log_post_m = log_post_m_cand
            accepted += 1

    ensembles[i] = ensemble
    acc_rates[i] = accepted / proposed
    autocorrs[i] = emcee.autocorr.integrated_time(ensemble, quiet=True).mean().item()

utils.psave(ensembles, 'data/proposal_trade_off_ensembles.obj')
utils.psave(acc_rates, 'data/proposal_trade_off_acc_rates.obj')
utils.psave(autocorrs, 'data/proposal_trade_off_autocorrs.obj')
utils.psave(widths, 'data/proposal_trade_off_widths.obj')
