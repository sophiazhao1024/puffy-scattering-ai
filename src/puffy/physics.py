"""Independent finite-ball Yukawa reference, NOT a literal notebook port.

Natural units. t=m_phi*r, y_i=m_phi*R_i, kappa=mu*v/m_phi,
b=2*mu*alpha/m_phi. V=-alpha*m_phi*U for attraction.
Normalized uniform charge densities; no extra N or A enhancement is implicit.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import spherical_jn, spherical_yn
from numpy.polynomial.legendre import leggauss


def overlap(s, a, b):
    """Overlap volume of spheres of radii a,b with centre separation s."""
    s=np.asarray(s,dtype=float)
    v=np.full_like(s,4*np.pi*min(a,b)**3/3)
    mask=s>abs(a-b)
    z=s[mask]
    v[mask]=np.pi*(a+b-z)**2*(z*z+2*z*(a+b)-3*(a-b)**2)/(12*z)
    return np.where(s>=a+b,0,v)


@lru_cache(maxsize=8)
def quadrature(n):
    return leggauss(n)


def ball_u(t,a,b,n=48):
    """Positive Yukawa convolution via exact overlap geometry and quadrature.

    Integration splits at |a-b| and t. No equal-radius 2R switch is assumed.
    Supports 0.05<=a,b<=5 and t>=0. Accuracy is checked only at test/demo points.
    """
    if not np.isfinite([t,a,b]).all() or t<0 or not (0.05<=a<=5 and 0.05<=b<=5):
        raise ValueError('Require t>=0 and radii in [0.05,5]')
    nodes,weights=quadrature(n)
    cuts=sorted(set([0.,abs(a-b),a+b]+([t] if 0<t<a+b else [])))
    total=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        s=lo+(nodes+1)*(hi-lo)/2
        rho=overlap(s,a,b)/((4*np.pi*a**3/3)*(4*np.pi*b**3/3))
        if t==0:
            kernel=4*np.pi*s*np.exp(-s)
        else:
            # exp(-|t-s|)-exp(-t-s), evaluated without cancellation.
            kernel=2*np.pi*s/t*np.exp(-np.abs(t-s))*(-np.expm1(-2*np.minimum(t,s)))
        total+=(hi-lo)/2*np.dot(weights,rho*kernel)
    return float(total)


def point_ball_u(t,y):
    """Analytic one point/one uniform ball reference; y>0, moderate y."""
    if t<0 or not 0<y<=5: raise ValueError('Invalid distance or radius')
    if t==0:return 3/y**3*(1-(1+y)*np.exp(-y))
    if t<y:return 3/y**3*(1-(1+y)*np.exp(-y)*np.sinh(t)/t)
    return 3*(y*np.cosh(y)-np.sinh(y))/y**3*np.exp(-t)/t


@dataclass(frozen=True)
class Config:
    lmax:int=10
    rtol:float=2e-8
    atol:float=2e-10
    grid:int=700
    tail:float=22.
    start:float=1e-4
    quadrature:int=48


def phase_shift(potential,kappa,b,l,end,cfg):
    """Integrate u=t^(l+1)w, then match to SPHERICAL Bessel functions.

    Cross-multiplied matching avoids division by u at nodes. Phase is modulo pi.
    """
    t0=cfg.start
    q0=kappa*kappa+b*float(potential(0.))
    w0=1-q0*t0*t0/(2*(2*l+3))
    wp0=-q0*t0/(2*l+3)
    def ode(t,z):
        return [z[1],-2*(l+1)*z[1]/t-(kappa*kappa+b*float(potential(t)))*z[0]]
    sol=solve_ivp(ode,(t0,end),(w0,wp0),method='DOP853',rtol=cfg.rtol,atol=cfg.atol,
                  max_step=min(.35,.4/kappa))
    if not sol.success or not np.isfinite(sol.y[:,-1]).all():
        raise RuntimeError('ODE failed: '+sol.message)
    w,wp=sol.y[:,-1];z=kappa*end;B=l*w+end*wp
    num=z*spherical_jn(l,z,True)*w-B*spherical_jn(l,z)
    den=z*spherical_yn(l,z,True)*w-B*spherical_yn(l,z)
    return float(np.arctan2(num,den))


def solve(b,kappa=.35,ychi=1.,yn=.5,cfg=Config()):
    """Return S=sigma_T*k^2/(4*pi), phase shifts and truncation diagnostics."""
    if not np.isfinite([b,kappa,ychi,yn]).all() or b<0 or kappa<=0:
        raise ValueError('Require finite b>=0,kappa>0')
    if not isinstance(cfg.lmax,int) or cfg.lmax<3 or cfg.grid<100 or cfg.start<=0 or cfg.tail<10:
        raise ValueError('Invalid solver configuration')
    if not (.05<=ychi<=5 and .05<=yn<=5):raise ValueError('Radii outside supported input range')
    end=ychi+yn+cfg.tail
    # Exact zero-coupling solution: a relative tail is undefined for zero S.
    if b == 0:
        return {'S': 0., 'sigma_mphi2': 0., 'phases': [0.]*(cfg.lmax+2),
                'tail_fraction': 0., 'end': end}
    grid=np.unique(np.r_[np.linspace(0,end,cfg.grid),abs(ychi-yn),ychi+yn])
    pot=PchipInterpolator(grid,[ball_u(t,ychi,yn,cfg.quadrature) for t in grid])
    phases=np.array([phase_shift(pot,kappa,b,l,end,cfg) for l in range(cfg.lmax+2)])
    terms=np.arange(1,cfg.lmax+2)*np.sin(np.diff(phases))**2
    S=float(terms.sum())
    tail_ratio=float(terms[-3:].sum()/max(S,1e-30))
    if tail_ratio>1e-5:raise RuntimeError(f'Partial-wave tail unresolved: {tail_ratio}')
    return {'S':S,'sigma_mphi2':4*np.pi*S/kappa**2,'phases':phases.tolist(),
            'tail_fraction':tail_ratio,'end':end}


def form_factor(z):
    z=np.asarray(z,float);small=np.abs(z)<1e-3
    safe=np.where(small,1.,z)
    ans=3*(np.sin(safe)-safe*np.cos(safe))/safe**3
    return np.where(small,1-z*z/10+z**4/280,ans)


def born_S(b,kappa,ychi,yn):
    """Independent momentum-space Born transfer cross section in same units."""
    c,w=leggauss(160);q=kappa*np.sqrt(2*(1-c))
    amp=b*form_factor(q*ychi)*form_factor(q*yn)/(1+q*q)
    return float(kappa*kappa/2*np.dot(w,(1-c)*amp**2))


def dimensional(mu,alpha,mphi,v,Rchi,Rn):
    if min(mu,alpha,mphi,v,Rchi,Rn)<=0:raise ValueError('All inputs must be positive')
    return dict(b=2*mu*alpha/mphi,kappa=mu*v/mphi,ychi=mphi*Rchi,yn=mphi*Rn)


def sigma_cm2(S,mu,v):
    """1 GeV^-2 = 0.3893793721e-27 cm^2."""
    if S<0 or mu<=0 or v<=0:raise ValueError('Invalid conversion inputs')
    return 4*np.pi*S/(mu*v)**2*0.3893793721e-27
