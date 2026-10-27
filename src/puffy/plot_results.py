"""Optional plots from saved NEW prototype outputs; no new physics labels.

python -m pip install -e '.[plot]'
python -m puffy.plot_results results/demo
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from .surrogate import Surrogate, interpolation


def main(folder='results/demo'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11})
    folder=Path(folder)
    with (folder/'training.csv').open() as f: train=list(csv.DictReader(f))
    with (folder/'holdout.csv').open() as f: test=list(csv.DictReader(f))
    report=json.loads((folder/'metrics.json').read_text())
    b=np.array([float(r['b']) for r in train]);S=np.array([float(r['S']) for r in train])
    active=np.array([r['split']=='active_train' for r in train])
    bt=np.array([float(r['b']) for r in test]);st=np.array([float(r['S']) for r in test])
    x=np.geomspace(b.min(),b.max(),400);model=Surrogate.fit(b,S,report['seed']);gp=model.predict(x)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    ax=axes[0,0]
    ax.loglog(x,gp['median'],label='Final GP',color='#166a83')
    ax.loglog(x,interpolation(b,S,x),'--',label='Final PCHIP',color='#be6f21',alpha=.8)
    ax.scatter(bt,st,color='black',s=24,marker='x',label='Holdout reference',zorder=5)
    ax.set(xlabel='b',ylabel='S = sigma_T k^2 / (4 pi)',title='Independent finite-ball solver labels')
    ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.scatter(b[~active],S[~active],label=f'Initial train ({(~active).sum()})',s=25,color='#166a83')
    ax.scatter(b[active],S[active],label=f'Active train ({active.sum()})',s=65,marker='*',color='#be6f21')
    ax.set(xscale='log',yscale='log',xlabel='b',ylabel='S',title='Training locations; no test-label selection')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    for key,label,color in [('gp_final','Final GP','#166a83'),('pchip_final','Final PCHIP','#be6f21'),('born','Born','#7b7490')]:
        p=np.array([float(r[key]) for r in test]);ax.scatter(bt,np.maximum(abs(p-st)/st,1e-10),label=label,color=color,s=26)
    ax.set(xscale='log',yscale='log',xlabel='b',ylabel='Absolute relative error',title='Untouched holdout: 20 points')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for key,label in [('GP_initial','GP: initial'),('GP_final','GP: final'),('PCHIP_initial','PCHIP: initial'),('PCHIP_final','PCHIP: final')]:
        ax.barh(label,report['metrics'][key]['log10_rmse'],color='#166a83' if key.startswith('GP') else '#be6f21')
    ax.set(xlabel='log10 RMSE',title='Same fixed test set; lower is better')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.xaxis.set_major_locator(FixedLocator([.2,.5,1,2,5,7]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos:f'{val:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes.flat:ax.grid(alpha=.18)
    fig.suptitle('NEW AI prototype: fixed scattering slice\nkappa=0.35, y_chi=1, y_n=0.5 | independent solver labels',fontsize=12)
    fig.savefig(folder/'benchmark.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'results/demo')
