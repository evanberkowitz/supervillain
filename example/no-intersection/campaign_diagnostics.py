#!/usr/bin/env python

r"""
Per-ensemble diagnostic PDFs for a campaign.py output directory: next to each
``N{N}/kappa{κ}.h5`` a ``kappa{κ}.pdf`` with

* histories, measurement histograms, and bootstrap bands/histograms of the scalar
  observables (:func:`supervillain.analysis.comparison_plot` machinery, one ensemble);
* bootstrap distributions of the derived scalars (χ_S, χ_θ, the Binder U);
* :meth:`~.Bootstrap.plot_correlator` pages for ``Spin_Spin_Normalized`` and the
  absolutely-normalized Θ correlator (``Theta_Theta`` over ``Vacuum_Ticks`` per draw,
  with the identically-1 contact value restored at the origin).

Everything is reconstructed from the h5 alone (the bootstrap group carries its
decorrelated ensemble), so this can run long after the campaign.

Run from example/no-intersection/:

    uv run python campaign_diagnostics.py --outdir campaign-2026-07-08
    uv run python campaign_diagnostics.py --only campaign-2026-07-08/N6/kappa0.1.h5
"""

import argparse
import glob
import os
import traceback

import h5py as h5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import supervillain.analysis
from supervillain.analysis import Bootstrap, Uncertain
import supervillain.analysis.comparison_plot as comparison_plot

HISTORIES = ('ActionDensity', 'InternalEnergyDensity', 'InternalEnergyDensitySquared',
             'WindingSquared', 'Vacuum_Ticks')
DERIVED = ('SpinSusceptibility', 'IntersectionSusceptibility', 'ThetaBinderCumulant')


def _histories(ax, e, observables):
    # comparison_plot.histories with the τ computation guarded: deep in a phase an
    # observable can be exactly constant (e.g. WindingSquared at large κ), where the
    # autocorrelation estimator raises rather than reporting nonsense.
    for a, o in zip(ax, observables):
        try:
            tau = supervillain.analysis.autocorrelation_time(np.asarray(getattr(e, o)).real)
            label = f'τ={tau}'
        except ValueError:
            label = 'τ undefined (no fluctuations)'
        e.plot_history(a, o, alpha=0.5,
                       history_kwargs={'zorder': -1, 'label': label})
        a[0].legend(loc='upper left')


def diagnose(h5path, pdfpath=None):
    r"""Write the diagnostic PDF next to ``h5path``; returns the PDF path."""
    if pdfpath is None:
        pdfpath = os.path.splitext(h5path)[0] + '.pdf'
    with h5.File(h5path, 'r') as f:
        b = Bootstrap.from_h5(f['bootstrap'])
        attrs = dict(f.attrs)
    e = b.Ensemble          # the decorrelated ensemble rides inside the bootstrap
    title = (f"N={attrs['N']} κ={attrs['kappa']:g} ζ={attrs['zeta']:g} "
             f"configurations={attrs['configurations']} τ={attrs['tau']} "
             f"decorrelated={attrs['decorrelated']}")

    with PdfPages(pdfpath) as pdf:
        # Histories + bootstrap bands and histograms of the scalar observables.  The
        # stored ensemble is already decorrelated, so the quoted τ should be ~1.
        fig, ax = comparison_plot.setup(HISTORIES)
        comparison_plot.bootstraps(ax, (b,), ('',), observables=HISTORIES)
        _histories(ax, e, HISTORIES)
        fig.suptitle(title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Bootstrap distributions of the derived scalars.
        fig, axs = plt.subplots(1, len(DERIVED), figsize=(4 * len(DERIVED), 3.2))
        for a, name in zip(axs, DERIVED):
            dist = np.asarray(getattr(b, name)).real
            a.hist(dist, bins=25, density=True, alpha=0.7)
            a.set_xlabel(name)
            a.set_title(str(Uncertain(dist.mean(), dist.std())))
        fig.suptitle(title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Correlators.  The physical Θ is the per-draw ratio of the inline dwell
        # histogram to the vacuum clock; Θ(0) = 1 identically (a coincident pair IS
        # the vacuum) but the origin bin is structurally empty, so restore the known
        # contact value by hand to carry the absolute scale.
        Theta = (np.asarray(b.Theta_Theta).real
                 / np.asarray(b.Vacuum_Ticks).real[:, None, None, None, None])
        Theta[(slice(None),) + (0,) * (Theta.ndim - 1)] = 1.
        b.__dict__['Theta'] = Theta
        correlators = (('Spin_Spin_Normalized', 'log'), ('Theta', 'log'))
        fig, axs = plt.subplots(len(correlators), 1, sharex=True, squeeze=False,
                                figsize=(6, 3 * len(correlators)))
        axs = axs[:, 0]
        # Linear Δx (not the log of action-comparison.py) so the Δx = 0 point ---
        # Θ's absolute anchor Θ(0) = 1 --- is visible.
        for a, (name, yscale) in zip(axs, correlators):
            b.plot_correlator(a, name)
            a.set_yscale(yscale)
            a.set_ylabel(name)
        for a in axs[:-1]:
            a.set_xlabel('')
        axs[-1].set_xlabel('Δx')
        fig.suptitle(title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return pdfpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-ensemble campaign diagnostics.')
    parser.add_argument('--outdir', default='campaign-2026-07-08')
    parser.add_argument('--only', default=None, help='diagnose a single h5 file')
    parser.add_argument('--force', action='store_true',
                        help='regenerate PDFs that already exist')
    args = parser.parse_args()

    paths = [args.only] if args.only else sorted(glob.glob(f'{args.outdir}/N*/kappa*.h5'))
    for path in paths:
        pdfpath = os.path.splitext(path)[0] + '.pdf'
        if not args.force and args.only is None and os.path.exists(pdfpath):
            print(f'{pdfpath} exists, skipping', flush=True)
            continue
        try:
            print(f'wrote {diagnose(path)}', flush=True)
        except Exception:
            print(f'{path} FAILED', flush=True)
            traceback.print_exc()
