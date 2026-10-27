"""NEW experiment: guarded Gaussian-process interpolation on ONE fixed slice.

Labels are our independent solver's output, not the paper's data or notebook
exports. All machine learning uses log10(b), log10(S). There is no claim that
GP is superior to interpolation or that posterior variance certifies physics.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class Surrogate:
    gp: object
    bounds: tuple
    warnings: list

    @classmethod
    def fit(cls, b, S, seed=2026):
        b=np.asarray(b, dtype=float); S=np.asarray(S, dtype=float)
        if b.ndim!=1 or b.size<4 or b.shape!=S.shape:
            raise ValueError('Need matching 1D arrays with at least four points')
        if not (np.isfinite(b).all() and np.isfinite(S).all() and (b>0).all() and (S>0).all()):
            raise ValueError('Positive finite b and S required')
        if np.unique(b).size!=b.size: raise ValueError('Duplicate training points')
        order=np.argsort(b);b=b[order];S=S[order]
        kernel=ConstantKernel(1.,(1e-3,1e3))*Matern(.3,(.015,3.),nu=2.5)+WhiteKernel(1e-8,(1e-10,1e-3))
        gp=GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,random_state=seed,alpha=1e-10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            gp.fit(np.log10(b)[:,None],np.log10(S))
        return cls(gp,(float(b[0]),float(b[-1])),[str(w.message) for w in caught])

    def predict(self,b):
        b=np.atleast_1d(np.asarray(b,dtype=float))
        if not np.isfinite(b).all() or np.any(b<self.bounds[0]) or np.any(b>self.bounds[1]):
            raise ValueError('Extrapolation refused: call the reference solver')
        logmean,std=self.gp.predict(np.log10(b)[:,None],return_std=True)
        # 10**mean is the median in linear S, not the lognormal expectation.
        return {'median':10**logmean,'log10_mean':logmean,'log10_std':std,
                'lower95':10**(logmean-1.96*std),'upper95':10**(logmean+1.96*std)}


def metrics(actual,predicted):
    a=np.asarray(actual,float);p=np.asarray(predicted,float)
    if a.shape!=p.shape or (a<=0).any() or (p<=0).any() or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Matching positive finite metric inputs required')
    logerr=np.log10(p)-np.log10(a);relative=np.abs(p-a)/a
    return {'log10_rmse':float(np.sqrt(np.mean(logerr**2))),
            'median_relative_error':float(np.median(relative)),
            'p95_relative_error':float(np.quantile(relative,.95)),
            'max_relative_error':float(np.max(relative))}


def interpolation(b,S,query):
    order=np.argsort(b)
    pred=PchipInterpolator(np.log10(np.asarray(b)[order]),np.log10(np.asarray(S)[order]),extrapolate=False)(np.log10(query))
    if not np.isfinite(pred).all():raise ValueError('Interpolation outside training range')
    return 10**pred
