"""Reproducible NEW solver-generated data and untouched holdout benchmark."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import platform
import time
import numpy as np
import scipy
import sklearn
from .physics import Config, solve, born_S
from .surrogate import Surrogate, metrics, interpolation


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(out='results/my-run', ntrain=24, ntest=20, active=6, seed=2026):
    """One log-b slice only. Active candidates never access test labels."""
    if ntrain<8 or ntest<5 or active<0:raise ValueError('Need ntrain>=8, ntest>=5, active>=0')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);cfg=Config();lo,hi=.2,7.
    fixed={'kappa':.35,'ychi':1.,'yn':.5}
    btrain=np.geomspace(lo,hi,ntrain)
    # New random test locations are distinct from regular training and candidate grids.
    btest=10**rng.uniform(np.log10(lo),np.log10(hi),ntest)
    candidates=np.geomspace(lo,hi,501)
    rows=[];testrows=[];solve_seconds=0.
    def label(b,split):
        nonlocal solve_seconds
        start=time.perf_counter();res=solve(float(b),cfg=cfg,**fixed);dt=time.perf_counter()-start;solve_seconds+=dt
        row={'b':float(b),**fixed,'S':res['S'],'sigma_mphi2':res['sigma_mphi2'],
             'tail_fraction':res['tail_fraction'],'solver_seconds':dt,'split':split}
        (testrows if split=='test' else rows).append(row)
        return res['S']
    strain=np.array([label(b,'initial_train') for b in btrain])
    initial_b=btrain.copy();initial_S=strain.copy()
    start=time.perf_counter();model=Surrogate.fit(btrain,strain,seed);fit_seconds=time.perf_counter()-start
    initial_model=model;training_warnings=list(model.warnings)
    selected=[]
    for step in range(active):
        distances=np.min(abs(np.log10(candidates[:,None])-np.log10(btrain)),axis=1)
        pool=candidates[distances>1e-7]
        if not len(pool):raise ValueError('Active candidate pool exhausted')
        pred=model.predict(pool);next_b=float(pool[np.argmax(pred['log10_std'])]);selected.append(next_b)
        next_S=label(next_b,'active_train')
        btrain=np.append(btrain,next_b);strain=np.append(strain,next_S)
        start=time.perf_counter();model=Surrogate.fit(btrain,strain,seed);fit_seconds+=time.perf_counter()-start
        training_warnings.extend(model.warnings)
    # Test labels are generated only after all adaptive model decisions are fixed.
    actual=np.array([label(b,'test') for b in btest])
    start=time.perf_counter();pred=model.predict(btest);prediction_seconds=time.perf_counter()-start
    initial_pred=initial_model.predict(btest)
    pchip=interpolation(btrain,strain,btest)
    pchip_initial=interpolation(initial_b,initial_S,btest)
    born=np.array([born_S(b,**fixed) for b in btest])
    coverage=float(np.mean((actual>=pred['lower95'])&(actual<=pred['upper95'])))
    for i,row in enumerate(testrows):
        row.update(gp_initial=float(initial_pred['median'][i]),gp_final=float(pred['median'][i]),
                   gp_lower95=float(pred['lower95'][i]),gp_upper95=float(pred['upper95'][i]),
                   pchip_initial=float(pchip_initial[i]),pchip_final=float(pchip[i]),born=float(born[i]))
    test_solver_time=sum(row['solver_seconds'] for row in testrows)
    report={'provenance':'NEW prototype, independent standard solver; not paper reproduction or notebook export',
            'seed':seed,'domain':{'b':[lo,hi],**fixed},'config':asdict(cfg),
            'n_initial_train':ntrain,'n_active':active,'n_test':ntest,'selected_b':selected,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
            'metrics':{'GP_initial':metrics(actual,initial_pred['median']),'GP_final':metrics(actual,pred['median']),
                       'PCHIP_initial':metrics(actual,pchip_initial),'PCHIP_final':metrics(actual,pchip),'Born':metrics(actual,born)},
            'gp_95_interval_empirical_coverage':coverage,
            'timing':{'all_label_generation_seconds':solve_seconds,'all_gp_fits_seconds':fit_seconds,
                      'test_reference_seconds':test_solver_time,'gp_test_prediction_seconds':prediction_seconds,
                      'prediction_only_speedup':test_solver_time/max(prediction_seconds,1e-12)},
            'optimizer_warnings':sorted(set(training_warnings)),
            'limitations':['One fixed slice; no extrapolation; no detector likelihood or nuclear coherence factors.',
                           'Small random holdout is not a resonance-focused or out-of-distribution certification.',
                           'GP intervals are empirical model uncertainty, not rigorous numerical-error bounds.',
                           'Speedup excludes training and label generation; no end-to-end saving established.']}
    write_csv(out/'training.csv',rows);write_csv(out/'holdout.csv',testrows)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    # Independent numerical refinement at three non-training points; label fidelity only.
    finecfg=Config(lmax=14,grid=1400,tail=28,rtol=2e-10,atol=2e-12,start=5e-5,quadrature=64)
    refined=[]
    for b in [.37,2.15,5.6]:
        base=solve(b,**fixed,cfg=cfg)['S'];fine=solve(b,**fixed,cfg=finecfg)['S']
        refined.append({'b':b,'S_default':base,'S_refined':fine,'relative_difference':abs(base-fine)/fine})
    write_csv(out/'solver_refinement.csv',refined)
    return report
