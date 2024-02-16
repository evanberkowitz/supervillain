#!/usr/bin/env python

import matplotlib.pyplot as plt
import supervillain
from supervillain.analysis import Uncertain

_default_observables=('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared', 'SpinSusceptibility', 'WindingSquared')

def setup(observables=_default_observables):
    r'''

    The return values are the same as those of `matplotlib.pyplot.subplots <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_.

    Parameters
    ----------
        observables: iterable
            The observables you wish to compare.

    Returns
    -------
        fig: matplotlib.pyplot.figure
            A new figure for drawing comparisons.
        ax: array of axes
            Axes in the figure.  One row per observable.  Three columns, one for the Monte Carlo history, one for a histogram, and one for bootstraps.
            Even if setting up for only 1 observable, the array is two-dimensional.
    '''

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

    for a, o in zip(ax, observables):
        a[0].set_ylabel(o)

    return fig, ax

def bootstraps(ax, boots, labels=None, observables=_default_observables):
    r'''
    One row per observable, for each bootstrap object, calls :py:meth:`~.Ensemble.plot_history` on the underlying ensemble, :py:meth:`plot_band` on this history, and puts a bootstrap histogram in the third column.

    Parameters
    ----------
        ax: array of axes
        boots: iterable of Bootstraps
        labels: iterable of strings
        observables: iterable of strings
    '''
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

        a[2].legend()

def histories(ax, ensembles, labels=None, observables=_default_observables):
    r'''
    Calls :py:meth:`~.Ensemble.plot_history` for each observable and row in ax.
    Labels the trace with the corresponding label and the :py:func:`~.analysis.autocorrelation_time` of the observable.
    That makes it good for 'raw' ensembles; ensembles that are properly decorrelated will have a very short τ.

    Parameters
    ----------
        ax: array of axes
            As returned from :py:func:`~.setup()`.
        ensembles: iterable of Ensembles
            Each will have its Monte Carlo history plotted and histogrammed for each observable.
        labels: iterable of strings
            Names for the legend, one per ensemble.
        observables: iterable of strings
    '''
    for a, o in zip(ax, observables):
        for e, label in zip(ensembles,labels):
            tau = supervillain.analysis.autocorrelation_time(getattr(e, o))
            e.plot_history(a, o, alpha=0.5, 
                           history_kwargs={
                               'zorder': -1,
                               'label': f'{label} τ={tau}'
                            })
        a[0].legend()

