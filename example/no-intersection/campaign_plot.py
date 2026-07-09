#!/usr/bin/env python

r"""
Figures from a campaign.py output directory.

* ``punchline.pdf`` --- the anomaly-resolution figure: the spin susceptibility χ_S and
  the intersection susceptibility's excess χ_θ - 1 against κ, one curve per volume.
  Short-ranged phases show volume-independent constants; order shows χ ∝ V; a
  (sufficiently light) critical point shows volume growth in between.
* ``binder.pdf`` --- the Binder cumulant U(κ; L) with its exact symmetric-phase value
  2 - 1/V per volume and the broken-phase limit 1; curves at different volumes cross
  at a critical point with no knowledge of scaling dimensions.
* ``zeta.pdf`` --- the tuned fugacity ζ against κ per volume: the defect-pressure
  thermometer.  ζ dips where single-link moves light up charge most easily (dense
  sheets and the transition window) and shrinks with volume there; a ζ forced to
  scale like 1/V would itself be a defect-condensation (θ-order) diagnostic.

Run from example/no-intersection/:

    uv run python campaign_plot.py --outdir campaign-2026-07-08
"""

import argparse
import glob

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Campaign punchline + Binder figures.')
parser.add_argument('--outdir', default='campaign-2026-07-08')
args = parser.parse_args()


def collect(outdir):
    data = {}
    for path in sorted(glob.glob(f'{outdir}/N*/kappa*.h5')):
        with h5.File(path, 'r') as f:
            N = int(f.attrs['N'])
            row = {'kappa': float(f.attrs['kappa']),
                   'zeta': float(f.attrs['zeta']),
                   'suspect': bool(f.attrs.get('thermalization_suspect', False))}
            for name in ('SpinSusceptibility', 'IntersectionSusceptibility',
                         'ThetaBinderCumulant'):
                row[name] = (float(f[f'scalars/{name}/mean'][()]),
                             float(f[f'scalars/{name}/err'][()]))
            data.setdefault(N, []).append(row)
    for N in data:
        data[N].sort(key=lambda r: r['kappa'])
    return data


data = collect(args.outdir)
if not data:
    raise SystemExit(f'no h5 points found under {args.outdir}')

colors = {6: 'C0', 8: 'C1', 12: 'C2', 16: 'C3'}
marker = {False: 'o', True: 'x'}     # x marks thermalization-suspect points


def series(rows, key, transform=lambda m: m):
    k = np.array([r['kappa'] for r in rows])
    m = np.array([transform(r[key][0]) for r in rows])
    e = np.array([r[key][1] for r in rows])
    s = np.array([r['suspect'] for r in rows])
    return k, m, e, s


# ------------------------------------------------------------------- punchline
fig, (top, bot) = plt.subplots(2, 1, sharex=True, figsize=(7, 7),
                               gridspec_kw={'hspace': 0.05})
for N, rows in sorted(data.items()):
    c = colors.get(N, 'k')
    for suspect in (False, True):
        sel = [r for r in rows if r['suspect'] == suspect]
        if not sel:
            continue
        k, m, e, _ = series(sel, 'SpinSusceptibility')
        top.errorbar(k, m, yerr=e, color=c, marker=marker[suspect], ls='-' if not suspect else 'none',
                     label=f'N={N}' if not suspect else None)
        k, m, e, _ = series(sel, 'IntersectionSusceptibility', transform=lambda x: x - 1)
        bot.errorbar(k, m, yerr=e, color=c, marker=marker[suspect], ls='-' if not suspect else 'none')
top.set_yscale('log')
top.set_ylabel(r'$\chi_S$')
top.legend()
bot.set_xscale('log')
bot.set_yscale('log')
bot.set_xlabel(r'$\kappa$')
bot.set_ylabel(r'$\chi_\theta - 1$')
fig.suptitle('No-Intersection model: the two $U(1)$s against $\\kappa$')
fig.savefig(f'{args.outdir}/punchline.pdf', bbox_inches='tight')
fig.savefig(f'{args.outdir}/punchline.png', bbox_inches='tight', dpi=160)
print(f'wrote {args.outdir}/punchline.pdf')

# --------------------------------------------------------------------- binder
fig, ax = plt.subplots(figsize=(7, 4.5))
for N, rows in sorted(data.items()):
    c = colors.get(N, 'k')
    for suspect in (False, True):
        sel = [r for r in rows if r['suspect'] == suspect]
        if not sel:
            continue
        k, m, e, _ = series(sel, 'ThetaBinderCumulant')
        ax.errorbar(k, m, yerr=e, color=c, marker=marker[suspect],
                    ls='-' if not suspect else 'none',
                    label=f'N={N}' if not suspect else None)
    ax.axhline(2 - 1 / N**4, color=c, ls=':', lw=0.8)
ax.axhline(1, color='k', ls='--', lw=0.8)
ax.set_xscale('log')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'$U = \langle|M|^4\rangle / \langle|M|^2\rangle^2$')
ax.set_title('Binder cumulant of $M = \\sum_x e^{i\\theta_x}$ (dotted: $2 - 1/V$)')
ax.legend()
fig.savefig(f'{args.outdir}/binder.pdf', bbox_inches='tight')
fig.savefig(f'{args.outdir}/binder.png', bbox_inches='tight', dpi=160)
print(f'wrote {args.outdir}/binder.pdf')

# ----------------------------------------------------------------------- zeta
fig, ax = plt.subplots(figsize=(7, 4.5))
for N, rows in sorted(data.items()):
    c = colors.get(N, 'k')
    k = np.array([r['kappa'] for r in rows])
    z = np.array([r['zeta'] for r in rows])
    ax.plot(k, z, color=c, marker='o', label=f'N={N}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\kappa$')
ax.set_ylabel(r'tuned $\zeta$')
ax.set_title('Tuned fugacity (largest with vacuum dwell $> 15\\%$)')
ax.legend()
fig.savefig(f'{args.outdir}/zeta.pdf', bbox_inches='tight')
fig.savefig(f'{args.outdir}/zeta.png', bbox_inches='tight', dpi=160)
print(f'wrote {args.outdir}/zeta.pdf')
