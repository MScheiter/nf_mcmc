import numpy as np
import torch
from torch import nn

import tqdm

import neural_networks as nns


class RealNVP(nn.Module):
    def __init__(self, ndim, nflows, scale=1, gpu=True, gpu_id=0, parallel=False, **kwargs):
        # **kwargs are network parameters: nlayers, nhid, dropout
        super().__init__()

        if gpu:
            self.device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'
        print(self.device)


        self.ndim = ndim
        self.nflows = nflows

        self.scale = scale

        self.prior = torch.distributions.MultivariateNormal(torch.zeros(self.ndim).to(self.device), torch.eye(self.ndim).to(self.device))
        self.mask = self.get_mask()

        self.t = self.get_networks('t', **kwargs).to(self.device)
        self.s = self.get_networks('s', **kwargs).to(self.device)

    def get_mask(self):
        split = int(self.ndim / 2)

        row1 = torch.cat([torch.zeros(self.ndim - split), torch.ones(split)]).unsqueeze(dim=0)
        row2 = torch.cat([torch.ones(self.ndim - split), torch.zeros(split)]).unsqueeze(dim=0)
        mask = torch.cat([row1, row2], dim=0).repeat([self.nflows, 1]).to(self.device)

        return mask

    def get_networks(self, net_type, nlayers=3, nhid=256, dropout=0.4):
        # net type must be either 't' or 's'

        layers = [None] * nlayers

        for i in range(nlayers - 1):
            layers[i] = nns.create_layer_dict(self.ndim if i == 0 else nhid, nhid, normalize=False, dropout=dropout, activation='leakyrelu')
        layers[-1] = nns.create_layer_dict(nhid, self.ndim, normalize=False, dropout=None, activation='tanh' if net_type=='s' else None)

        module_list = nn.ModuleList([nns.mlp(layers) for _ in range(2 * self.nflows)])

        return module_list

    def g(self, z):
        # Forward operation: takes latent vector z and returns a sample x
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        # Inverse operation: takes a sample x and returns a sample z and the Jacobian
        log_det_J, z = x.new_zeros(x.shape[0]).to(self.device), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x, cpu=False):
        if type(x) == np.ndarray:
            x = torch.Tensor(x).unsqueeze(0)
        # Takes a sample x and returns its density (prob of latent vector plus Jacobian)
        z, logp = self.f(x.to(self.device) / self.scale)
        if cpu:
            return (self.prior.log_prob(z).to(self.device) + logp).cpu()
        else:
            return self.prior.log_prob(z).to(self.device) + logp

    def sample(self, N=1, dtype='numpy'):
        # Draws N samples from the generative model
        z = self.prior.sample((N, 1))
        # logp = self.prior.log_prob(z)
        x = self.g(z) * self.scale
        if dtype == 'numpy':
            return x.squeeze().detach().cpu().numpy()
        else:
            return x

    def optimize(self, lr, epochs, nbatch, data):

        if type(data) == np.ndarray:
            data = torch.Tensor(data)
        self.data = data
        self.epochs = epochs
        self.nbatch = nbatch
        self.lr = lr

        data = data.to(self.device)

        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad==True], lr=lr)
        nsamples = data.shape[0]
        iterations_per_epoch = int(nsamples / nbatch)

        self.losses = np.zeros(epochs)

        for j, epoch in tqdm.tqdm(enumerate(range(epochs))):

            data = data[torch.randperm(nsamples)]

            for i in range(iterations_per_epoch):

                data_batch = data[i*nbatch : (i+1)*nbatch]

                loss = -self.log_prob(data_batch).mean()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            self.losses[j] = loss
