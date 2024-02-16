#!/usr/bin/env python

import matplotlib.pyplot as plt
import supervillain
from supervillain.analysis import Uncertain

_default_observables=('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared', 'SpinSusceptibility', 'WindingSquared')

def setup(observables=_default_observables):
    fig, ax = plt.subplots(len(observables), 3,
        figsize=(12, 2.5*len(observables)),
        gridspec_kw={'width_ratios': [4, 1, 1], 'wspace': 0, 'hspace': 0},
        sharey='row',
        squeeze=False
    )

    ax[-1,0].set_xlabel('Monte Carlo time')
    ax[-1,1].set_xticks([])
    ax[-1,1].set_xlabel('Measurements')
    ax[-1,2].set_xticks([])
    ax[-1,2].set_xlabel('Bootstraps')

    return fig, ax

def bootstraps(ax, boots, labels=None, observables=_default_observables):

    if labels is None:
        labels = tuple('' for b in boots)

    for a, o in zip(ax, observables):
        for b, label in zip(boots, labels):
            b.Ensemble.plot_history(a, o,
                                    alpha=0.5,
            )
            b.plot_band(a[0], o)
            a[2].hist(getattr(b, o),
                density=True,
                orientation='horizontal', alpha=0.5, bins=25,
                label=f'{label} {Uncertain(*b.estimate(o))}'
            )


        a[0].set_ylabel(o)
        a[2].legend()

def histories(ax, ens, labels=None, observables=_default_observables):
    for a, o in zip(ax, observables):
        for e, label in zip(ens,labels):
            tau = supervillain.analysis.autocorrelation_time(getattr(e, o))
            e.plot_history(a, o, alpha=0.5, 
                           history_kwargs={
                               'zorder': -1,
                               'label': f'{label} τ={tau}'
                            })
        a[0].legend()

