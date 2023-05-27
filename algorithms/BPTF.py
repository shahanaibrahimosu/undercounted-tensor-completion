import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin

from path import Path
from argparse import ArgumentParser
import import_ipynb
from helper_functions.utils import *


def _gamma_bound_term(pa, pb, qa, qb):
    return sp.gammaln(qa) - pa * np.log(qb) + \
        (pa - qa) * sp.psi(qa) + qa * (1 - pb / qb)


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes=4, n_components=100,  max_iter=200, tol=0.0001,
                 smoothness=100, verbose=True, alpha=0.1, debug=False):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose
        self.debug = debug

        self.alpha = alpha                                      # shape hyperparameter
        self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        self.nz_recon_I = None
        self.log_nz_recon_I = None

    def _reconstruct_nz(self, subs_I_M):
        """Computes the reconstruction for only non-zero entries."""
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in range(self.n_modes):
            nz_recon_IK *= self.G_DK_M[m][subs_I_M[m], :]
        self.nz_recon_I = nz_recon_IK.sum(axis=1)
        return self.nz_recon_I

    def _log_reconstruct_nz(self, subs_I_M):
        """Computes the log reconstruction for only non-zero entries."""
        I = subs_I_M[0].size
        K = self.n_components
        log_nz_recon_IK = np.zeros((I, K))
        for m in range(self.n_modes):
            log_nz_recon_IK += np.log(self.G_DK_M[m][subs_I_M[m], :])

        self.log_nz_recon_I = sp.logsumexp(log_nz_recon_IK, axis=1)
        return self.log_nz_recon_I

    def _test_elbo(self, data):
        """Copies code from pmf.py.  Used for debugging."""
        assert data.ndim == 2
        if isinstance(data, skt.sptensor):
            X = data.toarray()
        else:
            X = np.array(data)
        Et = self.E_DK_M[0]
        Eb = self.E_DK_M[1].T
        Elogt = np.log(self.G_DK_M[0])
        Elogb = np.log(self.G_DK_M[1].T)
        gamma_t = self.gamma_DK_M[0]
        gamma_b = self.gamma_DK_M[1].T
        rho_t = self.delta_DK_M[0]
        rho_b = self.delta_DK_M[1].T
        Z = np.dot(np.exp(Elogt), np.exp(Elogb))
        bound = np.sum(X * np.log(Z) - Et.dot(Eb))
        a = self.alpha
        c = self.beta_M[0]
        bound += _gamma_bound_term(a, a * c, gamma_t, rho_t).sum()
        bound += self.n_components * X.shape[0] * a * np.log(c)
        bound += _gamma_bound_term(a, a, gamma_b, rho_b).sum()
        return bound

    def _elbo(self, data, mask=None):
        """Computes the Evidence Lower Bound (ELBO)."""

        if mask is None:
            uttkrp_K = self.sumE_MK.prod(axis=0)
        elif isinstance(mask, skt.dtensor):
            uttkrp_DK = mask.uttkrp(self.E_DK_M, 0)
            uttkrp_K = (self.E_DK_M[0] * uttkrp_DK).sum(axis=0)
        elif isinstance(mask, skt.sptensor):
            uttkrp_DK = sp_uttkrp(mask.vals, mask.subs, 0, self.E_DK_M)
            uttkrp_K = (self.E_DK_M[0] * uttkrp_DK).sum(axis=0)

        bound = -uttkrp_K.sum()

        if isinstance(data, skt.dtensor):
            subs_I_M = data.nonzero()
            vals_I = data[subs_I_M]
        elif isinstance(data, skt.sptensor):
            subs_I_M = data.subs
            vals_I = data.vals
        log_nz_recon_I = self._log_reconstruct_nz(subs_I_M)

        bound += (vals_I * log_nz_recon_I).sum()

        K = self.n_components
        for m in range(self.n_modes):
            bound += _gamma_bound_term(pa=self.alpha,
                                       pb=self.alpha * self.beta_M[m],
                                       qa=self.gamma_DK_M[m],
                                       qb=self.delta_DK_M[m]).sum()
            bound += K * self.mode_dims[m] * self.alpha * np.log(self.beta_M[m])
        return bound

    def _init_all_components(self, mode_dims):
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)

    def _init_component(self, m, dim):
        assert self.mode_dims[m] == dim
        K = self.n_components
        s = self.smoothness
        if not self.debug:
            gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
            delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        else:
            gamma_DK = s * np.ones((dim, K))
            delta_DK = s * np.ones((dim, K))
        self.gamma_DK_M[m] = gamma_DK
        self.delta_DK_M[m] = delta_DK
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))
        if m == 0 or not self.debug:
            self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
        if isinstance(data, skt.dtensor):
            tmp = data.astype(float)
            subs_I_M = data.nonzero()
            tmp[subs_I_M] /= self._reconstruct_nz(subs_I_M)
            uttkrp_DK = tmp.uttkrp(self.G_DK_M, m)

        elif isinstance(data, skt.sptensor):
            tmp = data.vals / self._reconstruct_nz(data.subs)
            uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M)

        self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

    def _update_delta(self, m, mask=None):
        if mask is None:
            self.sumE_MK[m, :] = 1.
            uttkrp_DK = self.sumE_MK.prod(axis=0)
        else:
            uttkrp_DK = mask.uttkrp(self.E_DK_M, m)
        self.delta_DK_M[m][:, :] = self.alpha * self.beta_M[m] + uttkrp_DK

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data, mask=None, modes=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        for m in range(self.n_modes):
            if m not in modes:
                self._clamp_component(m)

        if self.debug:
            curr_elbo = self._test_elbo(data)
        else:
            curr_elbo = self._elbo(data, mask=mask)
        if self.verbose:
            print ('ITERATION %d:\t'\
                  'Time: %f\t'\
                  'Objective: %.2f\t'\
                  'Change: %.5e\t'\
                % (0, 0.0, curr_elbo, np.nan))

        for itn in range(self.max_iter):
            s = time.time()
            for m in modes:
                self._update_gamma(m, data)
                self._update_delta(m, mask)
                self._update_cache(m)
                if m == 0 or not self.debug:
                    self._update_beta(m)  # must come after cache update!
                self._check_component(m)
            if self.debug:
                bound = self._test_elbo(data)
            else:
                bound = self._elbo(data, mask=mask)
            delta = (bound - curr_elbo) / abs(curr_elbo)
            e = time.time() - s
            if self.verbose:
                print ('ITERATION %d:\t'\
                      'Time: %f\t'\
                      'Objective: %.2f\t'\
                      'Change: %.5e\t'\
                      % (itn+1, e, bound, delta))
            if not (delta >= 0.0):
                raise Exception('\n\nNegative ELBO improvement: %e\n' % delta)
            curr_elbo = bound
            if delta < self.tol:
                break

    def set_component(self, m, E_DK, G_DK, gamma_DK, delta_DK):
        assert E_DK.shape[1] == self.n_components
        self.E_DK_M[m] = E_DK.copy()
        self.sumE_MK[m, :] = E_DK.sum(axis=0)
        self.G_DK_M[m] = G_DK.copy()
        self.gamma_DK_M[m] = gamma_DK.copy()
        self.delta_DK_M[m] = delta_DK.copy()
        self.beta_M[m] = 1. / E_DK.mean()

    def _clamp_component(self, m, version='geometric'):
        """Make a component a constant.
        This amounts to setting the expectations under the
        Q-distribution to be equal to a single point estimate.
        """
        assert (version == 'geometric') or (version == 'arithmetic')
        if version == 'geometric':
            self.E_DK_M[m][:, :] = self.G_DK_M[m]
        else:
            self.G_DK_M[m][:, :] = self.E_DK_M[m]
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def set_component_like(self, m, model, subs_D=None):
        assert model.n_modes == self.n_modes
        assert model.n_components == self.n_components
        D = model.E_DK_M[m].shape[0]
        if subs_D is None:
            subs_D = np.arange(D)
        assert min(subs_D) >= 0 and max(subs_D) < D
        E_DK = model.E_DK_M[m][subs_D, :].copy()
        G_DK = model.G_DK_M[m][subs_D, :].copy()
        gamma_DK = model.gamma_DK_M[m][subs_D, :].copy()
        delta_DK = model.delta_DK_M[m][subs_D, :].copy()
        self.set_component(m, E_DK, G_DK, gamma_DK, delta_DK)

    def fit(self, data, mask=None):
        assert data.ndim == self.n_modes
        data = preprocess(data)
        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        self._init_all_components(data.shape)
        self._update(data, mask=mask)
        return self

    def transform(self, modes, data, mask=None, version='geometric'):
        """Transform new data given a pre-trained model."""
        assert all(m in range(self.n_modes) for m in modes)
        assert (version == 'geometric') or (version == 'arithmetic')

        assert data.ndim == self.n_modes
        data = preprocess(data)
        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        self.mode_dims = data.shape
        for m, D in enumerate(self.mode_dims):
            if m not in modes:
                if self.E_DK_M[m].shape[0] != D:
                    raise ValueError('Pre-trained components dont match new data.')
            else:
                self._init_component(m, D)
        self._update(data, mask=mask, modes=modes)

        if version == 'geometric':
            return [self.G_DK_M[m] for m in modes]
        elif version == 'arithmetic':
            return [self.E_DK_M[m] for m in modes]

    def fit_transform(self, modes, data, mask=None, version='geometric'):
        assert all(m in range(self.n_modes) for m in modes)
        assert (version == 'geometric') or (version == 'arithmetic')

        self.fit(data, mask=mask)

        if version == 'geometric':
            return [self.G_DK_M[m] for m in modes]
        elif version == 'arithmetic':
            return [self.E_DK_M[m] for m in modes]

    def reconstruct(self, mask=None, version='geometric', drop_diag=False):
        """Reconstruct data using point estimates of latent factors.
        Currently supported only up to 5-way tensors.
        """
        assert (version == 'geometric') or (version == 'arithmetic')
        if version == 'geometric':
            tmp = [G_DK.copy() for G_DK in self.G_DK_M]
        elif version == 'arithmetic':
            tmp = [E_DK.copy() for E_DK in self.E_DK_M]

        Y_pred = parafac(tmp)
        if drop_diag:
            diag_idx = np.identity(Y_pred.shape[0]).astype(bool)
            Y_pred[diag_idx] = 0
        return Y_pred
