import numpy as np
import pytest
from puffy.surrogate import Surrogate, metrics, interpolation


def test_positive_interpolation_and_bounds():
    b=np.geomspace(.2,7,16);S=b*b
    gp=Surrogate.fit(b,S);query=np.geomspace(.22,6.9,25);p=gp.predict(query)
    assert (p['lower95']>0).all()
    assert (p['lower95']<=p['median']).all()
    assert (p['median']<=p['upper95']).all()
    assert metrics(query**2,p['median'])['log10_rmse']<.015
    with pytest.raises(ValueError):gp.predict([.1])
    with pytest.raises(ValueError):gp.predict([8.])
    with pytest.raises(ValueError):gp.predict([np.nan])


def test_baseline_power_law():
    b=np.geomspace(.2,7,10);q=np.geomspace(.21,6.9,19)
    np.testing.assert_allclose(interpolation(b,b*b,q),q*q,rtol=1e-12)
    with pytest.raises(ValueError):interpolation(b,b*b,[8])


def test_metrics_and_bad_inputs():
    assert metrics([1,2],[1,2])['max_relative_error']==0
    assert metrics([1],[10])['log10_rmse']==1
    with pytest.raises(ValueError):Surrogate.fit([1,2,3,4],[1,2,0,4])
    with pytest.raises(ValueError):Surrogate.fit([1,2,2,4],[1,2,3,4])
    with pytest.raises(ValueError):metrics([0],[1])
