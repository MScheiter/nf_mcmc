import numpy as np
from scipy.stats import multivariate_normal

class Circle:
    def __init__(self, ndim=2, d_mean=0.7, d_std=0.1, a=-1, b=1, nbins=100):
        self.d_mean = d_mean
        self.d_std = d_std
        self.a = a
        self.b = b
        self.nbins = nbins
        self.ndim = ndim

    def log_post(self, m):
        return self.log_prior(m) + self.log_likelihood(m)

    def log_prior(self, m):
        for i in range(len(m)):
            if m[i] < self.a or m[i] > self.b:
                return -np.inf
        return 0.0

    def log_likelihood(self, m):
        return -0.5*np.sum(((self.d_mean-self.forward(m))/self.d_std)**2)

    def forward(self, m):
        if type(m)==tuple:
            m = np.array(m)
        return np.linalg.norm(np.array(m), axis=0)

class LayeredModel:
    def __init__(self, noise_amp=0.1, noise_corr=0.01, nlayers=10, nt=231, nstations=7,
                    z_lims=(0, 10), vp_lims=(4.5, 8), fixed_z=False, lvz=False):

        self.noise_amp = noise_amp
        self.noise_corr = noise_corr
        self.nlayers = nlayers
        self.nt = nt
        self.nstations = nstations

        self.fixed_z = fixed_z
        self.lvz = lvz

        self.ndim = self.nlayers if self.fixed_z else 2*self.nlayers - 1

        self.z_lims = z_lims
        self.vp_lims = vp_lims

        self.a = np.concatenate([self.z_lims[0] * np.ones(nlayers-1), self.vp_lims[0] * np.ones(nlayers)])
        self.b = np.concatenate([self.z_lims[1] * np.ones(nlayers-1), self.vp_lims[1] * np.ones(nlayers)])

        self.cov = self.get_cov_matrix(noise_corr, nt)

        self.true_model = self.get_model()

        self.tt, self.obs_data = self.forward(self.true_model)

    def get_cov_matrix(self, corr, nx):
        k = lambda x, xp: np.exp(-(x-xp)**2 / (2.*corr**2))
        xx = np.linspace(-1, 1, nx)

        cov = np.zeros([nx, nx])
        for i in range(nx):
            for j in range(nx):
                cov[i, j] = k(xx[i], xx[j])
        return cov

    def get_random_noise(self):
        yy = np.random.multivariate_normal(np.zeros(self.nt), self.cov)
        return yy * self.noise_amp / np.std(yy)

    def get_model(self):
        z = np.random.rand(self.nlayers-1)  * (self.z_lims[1] - self.z_lims[0]) + self.z_lims[0]
        vp = np.random.rand(self.nlayers) * (self.vp_lims[1] - self.vp_lims[0]) + self.vp_lims[0]
        if not self.lvz:
            vp = np.sort(vp)
        return vp if self.fixed_z else np.concatenate([z, vp])

    def log_likelihood(self, model):
        data = self.forward(model, add_noise=False)[1]
        data_diff = data - self.obs_data
        ll = 0
        for k in range(self.nstations):
            ll += data_diff[k].dot(np.linalg.inv(self.cov).dot(data_diff[k].T))
            # print(ll)
        return -0.5 * ll

    def log_prior(self, model):
        for i in range(len(model)):
            if self.fixed_z:
                if model[i] < self.vp_lims[0] or model[i] > self.vp_lims[1]:
                    return -np.inf
            else:
                if model[i] < self.a[i] or model[i] > self.b[i]:
                    return -np.inf
        return 0.0

    def log_post(self, model, normalized=True):
        # McMC algorithm gives models in box (0, 1), so here we need to rescale
        if normalized:
            if self.fixed_z:
                model = model * (self.vp_lims[1] - self.vp_lims[0]) + self.vp_lims[0]
            else:
                model = model * (self.b - self.a) + self.a
        log_p = self.log_prior(model)
        if log_p == 0.0:
            log_p += self.log_likelihood(model)
        return log_p

    def forward(self, model, zs=75, xs=5, ys=10, stf=lambda w: stf_ricker(w, 0.25), add_noise=True):

        if self.fixed_z:
            z = np.linspace(10, 50, self.nlayers-1)
            vp = model
            model = pp.LayeredStructureModel(np.column_stack([np.insert(z, 0, 0), vp, vp/1.7, 0.32*vp+0.77]), interface_depth_form=True)
        else:
            z, vp = model[:self.nlayers-1], model[self.nlayers-1:]
            model = pp.LayeredStructureModel(np.column_stack([np.insert(z, len(z), np.inf), vp, vp/1.7, 0.32*vp+0.77]), interface_depth_form=False)

        xr = np.zeros(self.nstations)
        yr = np.linspace(10, 50, self.nstations)

        strike = 90
        dip = 45
        rake = -90
        scalar_moment = 3E7
        Mxyz = rtf2xyz(make_moment_tensor(strike, dip, rake, scalar_moment, 0, 0))
        F = np.zeros([3, 1])

        event_time = -10
        source =  pp.PointSource(xs, ys, zs, Mxyz, F, event_time)

        stations = pp.ListOfReceivers(xx=xr, yy=yr, depth=0)

        dt = 1/10
        tt, seis = pp.compute_seismograms(model, source, stations, self.nt, dt, alpha=0.1,
                                         xyz=True, source_time_function=stf, show_progress=False,
                                         stencil_kwargs={"kmin": 0, "kmax": 4.0, "nk": 1000},
                                         number_of_processes=100)

        seis = seis[:, 2, :]
        seis = seis / seis.std()
        if add_noise:
            for i in range(self.nstations):
                seis[i] += self.get_random_noise()

        return tt-event_time, seis
