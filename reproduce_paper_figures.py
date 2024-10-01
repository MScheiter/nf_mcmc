import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from PIL import Image
import torch

import emcee
import corner

import utils

plt.ion()
plt.close('all')


## Figure 1: Demonstration of trade-off for different proposals

plt.set_cmap('gist_heat_r')

fs = 16
plt.rcParams.update({'font.size': fs})

def add_label(ax, plot_number, leftshift=0.03, upshift=0):
    label = '(' + chr(96 + plot_number) + ')'
    ax.text(-0.05-leftshift, 1+upshift, label, color='k', transform=ax.transAxes, ha='right', va='top', fontweight='bold', fontsize=fs)

ensembles = utils.pload('data/proposal_trade_off_ensembles.obj')
acc_rates = utils.pload('data/proposal_trade_off_acc_rates.obj')
autocorrs = utils.pload('data/proposal_trade_off_autocorrs.obj')
widths = utils.pload('data/proposal_trade_off_widths.obj')

widths_plot_ind = [1, 5, 9, 13]
ls = 0.15

plt.close('all')
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

ax = axs[0+2]
ax.plot(widths, acc_rates, '--', color='darkorange', linewidth=3)
ax.plot(widths[widths_plot_ind], acc_rates[widths_plot_ind], '^', color='darkred', markersize=14)
ax.set_xscale('log')
ax.set_ylabel(r'Acceptance Rate $r_a$')
ax.set_xticks([])
ax.spines[['right', 'top']].set_visible(False)
add_label(ax, 5, ls)

ax = axs[3+2]
ax.plot(widths, autocorrs, '--', color='darkorange', linewidth=3)
ax.plot(widths[widths_plot_ind], autocorrs[widths_plot_ind], 'v', color='darkred', markersize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'Integr. Autocorr. Time $\tau$')
ax.set_xlabel(r'Proposal Width $\sigma$')
ax.spines[['right', 'top']].set_visible(False)

add_label(ax, 6, ls)

for i, ind in enumerate(widths_plot_ind):
    ax = axs[i if i<2 else i+1]
    ensemble = ensembles[ind]
    ax.hist2d(ensemble[:, 0], ensemble[:, 1], bins=np.linspace(-1, 1, 100))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if i == 5:
        ax.set_xticks()
    sigma = widths[ind]
    text = r'$\sigma$ = ' + str(np.round(sigma, 2 if i<3 else 1)) + '\n' + r'$r_a$ = ' + str(np.round(acc_rates[ind], 2)) + '\n' + r'$\tau$ = ' + str(int(autocorrs[ind]))
    ax.text(0, 0, text, va='center', ha='center', bbox=dict(facecolor='orange', alpha=0.9, ec='darkred', linewidth=3, boxstyle='round'))
    add_label(ax, i+1, -0.2)
    if i == 3:
        xxg = np.linspace(-0.3, 0.3, 100)
    else:
        xxg = np.linspace(-2*sigma, 2*sigma, 100)
    ax.plot(xxg, np.exp(-(xxg/sigma)**2)/np.sqrt(sigma)/20 + 0.25, c='darkred', alpha=0.7, linewidth=3)
plt.tight_layout()

fig.savefig('figures/fig1_proposal_trade_off.png', dpi=300)


## Figure 2: Hypersphere McMC sampling with 3 different proposals

fs = 16
plt.rcParams.update({'font.size': fs})

def add_label(ax, plot_number, leftshift=0.03, upshift=0):
    label = '(' + chr(96 + plot_number) + ')'
    ax.text(-0.05-leftshift, 1+upshift, label, color='k', transform=ax.transAxes, ha='right', va='top', fontweight='bold', fontsize=fs)

colors = ['darkred', 'k', 'rebeccapurple', 'g']
names = ['Metropolis-Hastings', 'Affine-invariant', 'Normalizing Flow', 'Posterior']

margin = 0.4
ms = 11

fig = plt.figure(figsize=(15, 18))
axs = []

gs = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.1, figure=fig)
gs.update(left=0.1, right=0.97, top=0.995, bottom=0.815)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))

gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.16, figure=fig)
gs.update(left=0.1, right=0.97, top=0.76, bottom=0.545)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))
axs.append(plt.subplot(gs[2]))

gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.16, figure=fig)
gs.update(left=0.1, right=0.97, top=0.505, bottom=0.29)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))
axs.append(plt.subplot(gs[2]))

gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.16, figure=fig)
gs.update(left=0.1, right=0.97, top=0.25, bottom=0.035)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))
axs.append(plt.subplot(gs[2]))

# plot acceptance rate
ndims = 2, 4, 6, 10, 15, 20, 25
acc_rates = np.zeros((7, 3))

for j, ndim in enumerate(ndims):
    for i in range(2):
        sampler = utils.pload('data/'+'sampler_circle_'+str(ndim)+'_'+str(i)+'.obj')
        acc_rates[j, i] = sampler.acceptance_fraction.mean()

    history = utils.pload('data/nf_circle_mcmc_history_perc' + str(ndim) + '.obj')
    acc_rates[j, 2] = history.mean()

ax = axs[0]
ax.plot(ndims, acc_rates[:,0] ,'v--', color='darkred', label='Metropolis-Hastings', markersize=ms)
ax.plot(ndims, acc_rates[:,1] ,'k^--', label='Affine-invariant', markersize=ms)
ax.plot(ndims, acc_rates[:,2] ,'d--', c='rebeccapurple', label='Normalizing Flow', markersize=ms)
ax.plot(ndims, np.ones(len(ndims)), 'go--', label='Posterior', markersize=ms)
ax.set_xticks([5, 10, 15, 20, 25])

ax.set_yscale('log')
ax.set_xlabel('Dimensionality')
ax.set_ylabel('Acceptance Rate')
add_label(ax, 1, 0.1)

for i in range(4):
    ax.text(0.015, 0.01+i*0.06, names[i], color=colors[i], fontsize=fs, transform=ax.transAxes, ha='left', va='bottom')

# plot integrated autocorrelation times
calc_ac = False
if calc_ac:
    ac_times = np.zeros((7, 3))
    for j, ndim in enumerate(ndims):
        for i in range(2):
            sampler = utils.pload('data/'+'sampler_circle_'+str(ndim)+'_'+str(i)+'.obj')
            ac_times[j, i] = emcee.autocorr.integrated_time(sampler.get_chain(), quiet=True).mean().item()

        ensemble = utils.pload('data/nf_circle_mcmc_samples_perc' + str(ndim) + '.obj')
        ac_times[j, 2] = emcee.autocorr.integrated_time(ensemble, quiet=True).mean().item()
    utils.psave(ac_times, 'data/autocorr_circle.obj')
else:
    ac_times = utils.pload('data/autocorr_circle.obj')

ax = axs[1]
ax.plot(ndims, ac_times[:, 0] ,'v--', color='darkred', label='Metropolis-Hastings', markersize=ms)
ax.plot(ndims, ac_times[:, 1] ,'k^--', label='Affine-invariant', markersize=ms)
ax.plot(ndims, ac_times[:, 2] ,'d--', c='rebeccapurple', label='Normalizing Flow', markersize=ms)
ax.plot(ndims, np.ones(len(ndims)), 'go--', label='Posterior', markersize=ms)
ax.set_xticks([5, 10, 15, 20, 25])

ax.set_yscale('log')
ax.set_xlabel('Dimensionality')
ax.set_ylabel('Integr. Autocorr. Time')
add_label(ax, 2, 0.1)


# plot chain evolutions

for i, ndim in enumerate((2, 6, 10)):

    for j in range(3):
        if j < 2:
            sampler = utils.pload('data/'+'sampler_circle_'+str(ndim)+'_'+str(j)+'.obj')
            x = sampler.get_chain()[:,0,0]
        else:
            ensemble = utils.pload('data/nf_circle_mcmc_samples_perc' + str(ndim) + '.obj')
            x = ensemble[:, 0]

        ax = axs[j*3 + i + 2]
        ax.plot(x, c=colors[j])
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 10000)
        if i==2:
            if j==2:
                ax.set_xlabel('Sampling Step')
        ax.set_xticks([0, 2500, 5000, 7500, 10000])
        if i<2:
            ax.set_xticklabels([])
        ax.set_yticks([-1, 0, 1])

        descrip = str(ndim) + 'D'

        if i == 1:
            ax.set_ylabel('First Sampling Dimension')

        ax.text(0.015,0.9,descrip,color=colors[j],fontsize=fs,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.95,ec=colors[j],linewidth=0.8, boxstyle='round'))

        if i == 2:
            ax.text(0.985,0.1,names[j],color=colors[j],fontsize=fs,transform=ax.transAxes,ha='right',va='bottom',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.95,ec=colors[j],linewidth=0.8, boxstyle='round'))
        if i == 0:
            add_label(ax, j+3, -0.008)


fig.savefig('figures/fig2_hypersphere.png', dpi=300)



## Load emcee sampler for waveform example
sampler = utils.pload('data/emcee_pyprop8_lvz_long_long.obj')

## Figure 3: True Model and Noisy Data

def plot_one_waveform(data,tt,ax=None,style='k',shift=0,notime=False):
    if ax==None:
        fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(tt,data+shift,style)
    ax.set_xlim(tt.min(),tt.max())
    ax.set_yticks([])
    if notime:
        ax.set_xticks([])
    else:
        ax.set_xlabel('t [sec]')

def plot_step_model(model,zmax=80,ax=None,style='k',**kwargs):
    nlayers = int((len(model)+1)/2)
    z,vp = np.insert(model[:nlayers-1],0,0),model[nlayers-1:]
    if ax==None:
        fig, ax = plt.subplots(figsize=(3,6))
    for i in range(nlayers-1):
        ax.plot([vp[i],vp[i]],[z[i],z[i+1]],style)#,**kwargs)
        ax.plot([vp[i],vp[i+1]],[z[i+1],z[i+1]],style)#,**kwargs)
    ax.plot([vp[-1],vp[-1]],[z[-1],zmax],style,**kwargs)
    ax.set_xlim(4.4,8.1)
    ax.set_ylim(0,zmax)

    ax.set_xlabel(r'$v_p$ [km/s]')
    ax.set_ylabel('z [km]')
    ax.invert_yaxis()

    plt.tight_layout()


l = sampler.log_prob_fn.f.__self__
data = l.obs_data
vp = l.true_model
z = np.linspace(10, 50, l.nlayers-1)
model = np.concatenate([z, vp])

# plot true model and noisy data

shift_factor = 10

nrows, ncols = 1, 4

fig = plt.figure(figsize=(10, 5))
axs = []
axs.append(plt.subplot2grid((nrows, ncols), (0, 0)))
axs.append(plt.subplot2grid((nrows, ncols), (0, 1), colspan=3))

plot_step_model(model, ax=axs[0], style='k', zmax=60)
for k in range(7):
    plot_one_waveform(data[k], l.tt, ax=axs[1], style='k', shift=-shift_factor*k)#,notime=True)

fig.savefig('figures/fig3_true_model_data.png', dpi=300)

## Figure 4: Convergence Assessment

def add_label(ax, plot_number, leftshift=0.03, upshift=0):
    label = '(' + chr(96 + plot_number) + ')'
    ax.text(-0.05-leftshift, 1+upshift, label, color='k', transform=ax.transAxes, ha='right', va='top', fontweight='bold', fontsize=fs)

fs = 16
plt.rcParams.update({'font.size': fs})

log_p = sampler.get_log_prob()

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()

for i in range(2):
    axs[i].plot(-log_p)
    if i == 0:
        axs[i].add_patch(Rectangle((50000, 0), 50000, 40, linewidth=1, edgecolor='k', facecolor='none', zorder=20))
        axs[i].text(75000, 75, 'Zoom', va='center', ha='center')
        axs[i].plot([50000, 68500], [40, 75], 'k', linewidth=1)
        axs[i].plot([81500, 100000], [75, 40], 'k', linewidth=1)
        axs[i].set_ylim(0, 800)
        axs[i].set_xlim(0, 100000)
        axs[i].set_xticks([0, 25000, 50000, 75000, 100000])
        axs[i].set_yticks([0, 200, 400, 600, 800])
    else:
        axs[i].set_ylim(5, 40)
        axs[i].set_xlim(50000, 100000)
        axs[i].set_xticks([50000, 62500, 75000, 87500, 100000])
        axs[i].set_yticks([10, 20, 30, 40])
    axs[i].set_xlabel('Samples in chain')
    axs[i].set_ylabel('Negative log posterior')
plt.tight_layout()


data = sampler.get_chain()

idx = sampler.acceptance_fraction > 0.01

ns = np.linspace(0, 100000, 21, dtype='int')

autocorr = utils.pload('data/autocorr.obj')
autocorr_conv = utils.pload('data/autocorr_conv.obj')

for i in range(2):
    ax = axs[i+2]
    ac = autocorr if i == 0 else autocorr_conv
    label = 'All ' if i == 0 else 'Converged '
    label += 'chains'
    l1 = ax.plot(ns, ac[:, :-1], 'k--', alpha=0.5)
    l1 = ax.plot(ns, ac[:, -1], 'k--', alpha=0.5, label='Indiv. Parameters')
    l2 = ax.plot(ns, ac.mean(axis=1), 'k', linewidth=3, label='Average')
    if i == 0:
        ax.legend()
    ax.text(0.025,0.965,label,color='k',fontsize=fs,transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor=[0.9,0.9,0.9],alpha=0.95,ec='k',linewidth=0.8, boxstyle='round'))

    ax.set_xlabel('Samples in chain')
    ax.set_ylabel('Integr. Autocorr. Time')
    ax.set_xlim([0, 100000])
    ax.set_ylim([0, 7000 if i==0 else 6000])
    ax.set_xticks([0, 25000, 50000, 75000, 100000])
    ax.set_yticks([0, 1500, 3000, 4500, 6000])
plt.tight_layout()

for i, ax, in enumerate(axs):
    add_label(ax, i+1, leftshift=0.09, upshift=0.06)

fig.savefig('figures/fig4_waveform_convergence.png', dpi=300)



## Figure 5: Selected Marginals

fs = 14
plt.rcParams.update({'font.size': fs})

true_model = sampler.log_prob_fn.f.__self__.true_model

percentages = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
percentages = np.array([0.2, 1.0])
percentage = 0.2

fig, axs = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=False)#, layout='tight')

ind1s = 1, 3, 5, 7, 9
ind2s = 8, 6, 4, 2, 0

names = ['McMC', 'NF', 'NF-McMC']
colors = ['k', 'rebeccapurple', 'rebeccapurple']

for i in range(3):
    for j in range(5):
        ind1, ind2 = ind1s[j], ind2s[j]
        perc = int(100 * percentage)

        if i == 0:
            thinning = int(40 * percentage)
            n_samples = int(100000 * percentage)

            # Get McMC ensemble
            data = sampler.get_chain()[:n_samples:thinning].reshape(-1,10)
        elif i == 2:
            data = utils.pload('data/nf_pyprop8_mcmc_samples_perc' + str(perc) + '_step1.obj')
        else:
            data = utils.pload('data/nf_pyprop8_samples_perc' + str(perc) + '_step1.obj')[::2]
        data = data * 4.5 + 3.5
        x, y = data[:, ind1], data[:, ind2]
        ax = axs[i, j]
        corner.hist2d(x, y, ax=ax, range=[(3.5, 8)] * 2, color=colors[i], quiet=True)
        if i == 2:
            ax.set_xlabel(r'$v_p$ [km/s]')
        if j == 4:
            ax.set_ylabel(r'$v_p$ [km/s]')
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        else:
            ax.yaxis.set_ticklabels([])
        if j == 0:
            ax.set_ylabel(names[i], color=colors[i], fontsize=fs+3, bbox=dict(facecolor=[0.9, 0.9, 0.9], alpha=0.7, ec='k', linewidth=0.8, boxstyle='round'))

        if i == 0:
            txt = r'$m_{'+str(ind1+1)+'}$'+'$-$'+'$m_{'+str(ind2+1)+'}$'
            ax.set_title(txt, color='k', pad=15, fontsize=fs+3, bbox=dict(facecolor=[0.9, 0.9, 0.9], alpha=0.7, ec='k', linewidth=0.8, boxstyle='round'))

        ax.set_xticks([4, 5, 6, 7, 8])
        ax.set_yticks([4, 5, 6, 7, 8])
        ax.tick_params(bottom=True, top=True, right=True, left=True)#, labelleft=False, labelright=True)
        descrip = names[i]


plt.tight_layout()
plt.subplots_adjust(wspace=0.07, hspace=0.07)

fig.savefig('figures/fig5_waveform_densities.png', dpi=300)


## Figure 6: Step Plot

fs = 16
plt.rcParams.update({'font.size': fs})

def plot_step_model(model,zmax=80,ax=None,style='k',**kwargs):
    nlayers = int((len(model)+1)/2)
    z,vp = np.insert(model[:nlayers-1],0,0),model[nlayers-1:]
    if ax==None:
        fig, ax = plt.subplots(figsize=(3,6))
    for i in range(nlayers-1):
        ax.plot([vp[i],vp[i]],[z[i],z[i+1]],style)#,**kwargs)
        ax.plot([vp[i],vp[i+1]],[z[i+1],z[i+1]],style)#,**kwargs)
    ax.plot([vp[-1],vp[-1]],[z[-1],zmax],style,**kwargs)
    ax.set_xlim(4.4,8.1)
    ax.set_ylim(0,zmax)
    ax.set_xlabel(r'$v_p$ [km/s]')
    ax.set_ylabel('z [km]')
    ax.invert_yaxis()

    plt.tight_layout()

def get_step_profile(model, zmax=60):
    nlayers = l.nlayers

    vp = model
    z = np.linspace(10, 50, l.nlayers-1)
    z = np.tile(z, (100000, 1))

    depths = np.linspace(0, zmax, 1000)

    step_profile = np.zeros((vp.shape[0], 1000))

    i = 0
    for i in range(vp.shape[0]):
        step_profile[i] = vp[i,-1]
        for k in range(1,nlayers):
            step_profile[i,depths<z[i,-k]] = vp[i,-k-1]

    return step_profile, depths

l = sampler.log_prob_fn.f.__self__
data = l.obs_data
vp = l.true_model
z = np.linspace(10, 50, l.nlayers-1)
model = np.concatenate([z, vp])

### plot posterior density

for perc in 20,:

    thinning = int(perc * 2)
    n_samples = 10000
    titles = ['McMC', 'NF', 'NF-McMC']

    fig, axs = plt.subplots(1, 3, figsize=(12, 8))

    for i in range(3):
        if i == 0:
            ensemble = sampler.get_chain()[:perc*1000:thinning].reshape(-1,10) * 3.5 + 4.5 # get McMC ensemble
        elif i == 1:
            ensemble = utils.pload('data/nf_pyprop8_samples_perc' + str(perc) + '_step1.obj')[::10] * 3.5 + 4.5 # get NF ensemble
        else:
            ensemble = utils.pload('data/nf_pyprop8_mcmc_samples_perc' + str(perc) + '_step1.obj') * 3.5 + 4.5

        step_profile,depths = get_step_profile(ensemble)

        ax = axs[i]

        ax.hist2d(step_profile.flatten(),np.tile(depths,n_samples),bins=[np.linspace(4.5, 8, 30), np.linspace(0, 60, 1000)],cmap='binary')
        ax.invert_yaxis()
        plot_step_model(model, ax=ax, style='r', zmax=60, label='True model')
        ax.plot([-1, -1], [-1, -1], 'gray', label='Ensemble') # dummy line for legend
        median = np.concatenate([z, np.median(ensemble, axis=0)])
        plot_step_model(median, ax=ax, style='k--', zmax=60, label='Median')
        ax.set_title(titles[i])
        if i > 0:
            ax.set_ylabel(None)
        if i == 2:
            leg = ax.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    fig.savefig('figures/fig6_waveform_step_plot'+str(perc)+'.png', dpi=300)


## Figure 7: Acceptance Rate and Autocorrelation

fs = 16
plt.rcParams.update({'font.size': fs})

percentages = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

ars = np.zeros((5, 2))
acs = np.zeros((5, 2))

N = 10000

af = sampler.acceptance_fraction.mean()

for j in range(2):
    second_step = True if i==1 else False
    for k, percentage in enumerate(percentages):

        perc = int(100 * percentage)

        if j == 0:
            ensemble = utils.pload('data/nf_pyprop8_mcmc_samples_perc' + str(perc) + '_step1' + '.obj')
            history = utils.pload('data/nf_pyprop8_mcmc_history_perc' + str(perc) + '_step1' + '.obj')
        if j == 1:
            ensemble = utils.pload('data/nf_pyprop8_mcmc_samples_perc' + str(perc) + '_nextgen_step2_alt3' + '.obj')
            history = utils.pload('data/nf_pyprop8_mcmc_history_perc' + str(perc) + '_nextgen_step2_alt3' + '.obj')

        ars[k, j] = history.mean()
        acs[k, j] = emcee.autocorr.integrated_time(ensemble, quiet=True).item()


factor = 100000

fig = plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(percentages * factor, ars[:, 0], 'd--', c='rebeccapurple', label='NF Step 1', markersize=11)
plt.plot(percentages * factor, ars[:, 1], 'o--', c='rebeccapurple', label='NF Step 2', markersize=11)
plt.plot([percentages[0] * factor, percentages[-1] * factor], [af, af], 'k--', label='McMC')
plt.xlabel('Samples in chain')
plt.ylabel('Acceptance Rate')
plt.ylim([0, 0.5])
plt.xticks([20000, 60000, 100000])
plt.legend(loc='upper left', frameon=False)

ax = plt.subplot(1, 2, 2)
plt.plot(percentages * factor, acs[:, 0], 'd--', c='rebeccapurple', label='NF Step 1', markersize=11)
plt.plot(percentages * factor, acs[:, 1], 'o--', c='rebeccapurple', label='NF Step 2', markersize=11)
plt.plot([percentages[0] * factor, percentages[-1] * factor], [2676, 2676], 'k--', label='McMC')
ax.set_yscale('log')
plt.xlabel('Samples in chain')
plt.ylabel('Integr. Autocorr. Time')
plt.xticks([20000, 60000, 100000])
plt.ylim([1, 10000])

plt.tight_layout()

fig.savefig('figures/fig7_waveform_mcmc_2steps.png', dpi=300)


## Figure A1: Ricker wavelet and sketch

fs = 16
plt.rcParams.update({'font.size': fs})

def add_label(ax, plot_number, leftshift=0.03, upshift=0):
    label = '(' + chr(96 + plot_number) + ')'
    ax.text(-0.05-leftshift, 1+upshift, label, color='k', transform=ax.transAxes, ha='right', va='top', fontweight='bold', fontsize=fs)

def stf_ricker(omega, twidth):
    return 2*np.sqrt(2/3)*np.exp(-0.5*(twidth*omega)**2)*(np.pi**0.25)*(twidth**2.5)*(omega**2)

# Define source time function
twidth = 0.25
stf = lambda w: stf_ricker(w, twidth)

# fig, axs = plt.subplots(1, 3, figsize = (15, 4.2))
fig = plt.figure(figsize=(15, 4))
axs = []

gs = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.1, figure=fig)
gs.update(left=0.07, right=0.65, top=0.9, bottom=0.25)
axs.append(plt.subplot(gs[0]))
axs.append(plt.subplot(gs[1]))

gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.16, figure=fig)
gs.update(left=0.68, right=1, top=0.99, bottom=0.01)
axs.append(plt.subplot(gs[0]))


# include sketch of experimental setup
image = Image.open('figures/sketch_pyprop8.png')

ax = axs[2]
ax.imshow(image)
ax.axis('off')

# Plot Ricker wavelet in time domain
t = np.linspace(-1.5, 1.5, 1000)
y = 2 / (np.sqrt(3 * twidth) * np.pi**(1/4)) * (1 - (t/twidth)**2) * np.exp(-t**2 / (2 * twidth**2))

ax = axs[0]
ax.plot(t, y, 'k')
ax.set_xlabel('t [s]')
ax.set_ylabel('Amplitude')
ax.set_xlim(-1.5, 1.5)

# Plot power spectrum for Ricker wavelet
freqs = np.linspace(0, 2.5, 1000)       # Frequency in Hz
ps = np.abs(stf(2 * np.pi * freqs))**2

ax = axs[1]
ax.plot(freqs, ps, 'k')             # Square for power spectrum
ax.set_xlabel('f [Hz]')
ax.set_ylabel('Power Spectrum')
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 0.67)

for i, ax in enumerate(axs):
    add_label(ax, i+1, -0.04 if i==2 else 0.08, 0.02 if i==2 else 0.13)

fig.savefig('figures/figA1_ricker_stf_with_sketch.png', dpi=300)
