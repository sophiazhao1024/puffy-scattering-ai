import numpy as np
import pytest
from puffy.physics import *

def test_overlap_normalization():
    x,w=leggauss(180);a,b=.7,1.2
    cuts=[0,abs(a-b),a+b];norm=0
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        s=lo+(x+1)*(hi-lo)/2
        norm+=(hi-lo)/2*np.dot(w,4*np.pi*s*s*overlap(s,a,b)/((4*np.pi*a**3/3)*(4*np.pi*b**3/3)))
    assert norm==pytest.approx(1,abs=1e-12)

@pytest.mark.parametrize('t',[0,.3,1.,2.,6.])
def test_exchange_and_quadrature(t):
    assert ball_u(t,.7,1.2)==pytest.approx(ball_u(t,1.2,.7),rel=1e-12)
    assert ball_u(t,.7,1.2,32)==pytest.approx(ball_u(t,.7,1.2,80),rel=1e-10)

def test_point_ball_limit():
    for t in [0,.2,1.,3.]:
        assert ball_u(t,.05,1.)==pytest.approx(point_ball_u(t,1.),rel=.001)

def test_nonoverlap_exact():
    a,b=.7,1.2;t=4.
    h=lambda y:3*(y*np.cosh(y)-np.sinh(y))/y**3
    assert ball_u(t,a,b)==pytest.approx(h(a)*h(b)*np.exp(-t)/t,rel=1e-12)

def test_free_scattering():
    assert solve(0)['S']<1e-15

def test_weak_born():
    s=solve(.001)['S'];ref=born_S(.001,.35,1.,.5)
    assert s==pytest.approx(ref,rel=.003)

def test_convergence():
    base=solve(4.,ychi=.8)
    fine=solve(4.,ychi=.8,cfg=Config(lmax=14,rtol=2e-10,atol=2e-12,grid=1400,tail=28,start=5e-5,quadrature=64))
    assert base['S']==pytest.approx(fine['S'],rel=2e-4)

def test_s_wave_square_well():
    # Analytic tan(delta_0)=k/K*tan(KR) transformed by free propagation.
    k=.35;b=.8;R=1.3;cfg=Config()
    # Split exactly at the discontinuity by using a smooth analytic interior
    # matching instead of a numerical integration crossing the edge.
    d=phase_shift(lambda t:1.,k,b,0,R,cfg)
    K=np.sqrt(k*k+b)
    exact=np.arctan2(k*np.sin(K*R),K*np.cos(K*R))-k*R
    assert abs(np.sin(d-exact))<1e-7

@pytest.mark.parametrize('args',[dict(b=-1),dict(b=1,kappa=0),dict(b=1,ychi=20)])
def test_invalid(args):
    with pytest.raises(ValueError):solve(**args)

def test_units_and_form_factor():
    assert form_factor(0)==1
    assert sigma_cm2(1,1,1)==pytest.approx(4*np.pi*.3893793721e-27)
    assert dimensional(1,.1,.2,.001,5,2.5)['b']==1
