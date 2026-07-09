#!/usr/bin/env python

r"""
The ζ-censoring probe (campaign-2026-07-08/CENSORING.md diagnostics 1 + 2).

Two Generator-route runs at the same (N, κ) from the same thermalized valid start:

* **tuned** --- the campaign protocol: ``DefectGas.tune`` (vacuum dwell > 15%),
  D_max = 8, emit_every = 4V.
* **edge** --- ``DefectGas.tune_edge``: the largest ζ whose vacuum dwell stays above
  a small floor (the chain must still come home to emit), D_max raised, emit_every
  matched to the measured dwell.

Both runs carry the transport instrumentation (pair excursions, max separation,
excursion-length histogram), so every zero bin is reported as *censored with a
transport ceiling*, never as 0.000(0.000).  The mean is ζ-independent, so mid-range
disagreement beyond errors demonstrates finite-run transport bias (the two-ζ test);
agreement plus far bins lighting up at the edge pushes the censoring bound outward.

If a chain stops returning to the vacuum (the step RuntimeError), that is recorded
as the defect-condensation signature and a FRESH chain is started one ζ rung lower —
the result is never silently retried away.

Run from example/no-intersection/:

    uv run python censoring_probe.py --N 8 --kappa 0.02 --configurations 2000
"""

import argparse

import numpy as np

import supervillain
from supervillain.analysis import Bootstrap
from supervillain.lattice import Form, Lattice
import supervillain.generator.no_intersection as gen
import supervillain.generator.villain as villain
from supervillain.generator.combining import Sequentially

parser = argparse.ArgumentParser(description='ζ-censoring probe: tuned vs edge.')
parser.add_argument('--N', type=int, default=8)
parser.add_argument('--kappa', type=float, default=0.02)
parser.add_argument('--configurations', type=int, default=2000)
parser.add_argument('--therm', type=int, default=1500)
parser.add_argument('--floor', type=float, default=0.002)
parser.add_argument('--edge-D-max', type=int, default=32)
parser.add_argument('--seed', type=int, default=271828)
args = parser.parse_args()

L = Lattice(4, args.N)
S = supervillain.action.NoIntersections(L, kappa=args.kappa)
V = args.N**4
n_links = 4 * V


def defect_count(n):
    from supervillain.generator.no_intersection.charge import charge
    return int(np.abs(np.asarray(charge(Form(n, degree=1, lattice=L)))).sum())


def land_in_vacuum(phi, n, seed):
    for zc in (0.002, 0.0005):
        g = gen.DefectGas(S, zeta=zc, D_max=8, rng=np.random.default_rng(seed))
        for _ in range(400):
            if defect_count(n) == 0:
                return phi, n
            phi, n = g.run(phi, n, 5, tally=False)
    raise RuntimeError(f'cooldown failed: D = {defect_count(n)} persists')


def generate(label, zeta, D_max, emit_every, phi, n, seed):
    r"""One Generator-route run; a chain that stops returning to the vacuum is
    RECORDED as condensation-signature data and restarted fresh one rung lower."""
    ladder = (0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3)
    while True:
        gas = gen.DefectGas(S, zeta=zeta, D_max=D_max, emit_every=emit_every,
                            max_step_sweeps=2000, rng=np.random.default_rng(seed))
        chain = Sequentially((villain.SiteUpdate(S), villain.ExactUpdate(S),
                              villain.CohomologyUpdate(S), gas))
        try:
            e = supervillain.Ensemble(S).generate(
                args.configurations, chain,
                start={'phi': Form(phi, degree=0, lattice=L),
                       'n': Form(n, degree=1, lattice=L)})
            return e, gas, zeta
        except RuntimeError:
            below = [z for z in ladder if z < zeta]
            print(f'[{label}] RECORDED: zeta={zeta:g} stopped returning to the '
                  f'vacuum (defect-condensation signature; D_trace tail '
                  f'{gas.D_trace[-5:]}).  Fresh chain at zeta={below[-1]:g}.',
                  flush=True)
            zeta = below[-1]


def report(label, e, gas, zeta, emit_every):
    b = Bootstrap(e)
    T = np.asarray(b.Theta_Theta).real
    VT = np.asarray(b.Vacuum_Ticks).real
    Lsym = Lattice(4, args.N)
    Theta = Lsym.symmetrize(T / VT[:, None, None, None, None])
    mean, err = Theta.mean(axis=0), Theta.std(axis=0)

    counts = np.asarray(e.Theta_Theta).real.sum(axis=0) * V * zeta**2
    H_Z = np.asarray(e.Vacuum_Ticks).real.sum()
    one_count = 2.3 / (V * zeta**2 * H_Z)          # 90% Poisson upper limit scale

    exc = np.asarray(e.Pair_Excursions).real
    rmax = np.sqrt(np.asarray(e.Max_Pair_RSq).real.max())
    hist = np.asarray(e.Excursion_Lengths).real.sum(axis=0)
    top = int(np.max(np.nonzero(hist)[0])) if hist.sum() > 0 else 0

    chi = np.asarray(b.IntersectionSusceptibility).real
    U = np.asarray(b.ThetaBinderCumulant).real
    D_trace = np.array(gas.D_trace)

    print(f'\n=== {label}: zeta={zeta:g}  D_max={gas.D_max}  emit_every={emit_every}')
    print(f'  acceptance {gas.accepted/max(1, gas.proposed):.4f}   '
          f'pairs in flight D/2: mean {D_trace.mean()/2:.2f} max {int(D_trace.max())//2}')
    print(f'  vacuum dwell {H_Z/max(1, gas.proposed):.5f}   '
          f'excursions/step {exc.mean():.1f}   '
          f'max pair separation {rmax:.2f}   longest excursion < 2^{top} ticks')
    print(f'  chi_theta - 1 = {chi.mean()-1:+.4e} ({chi.std():.1e})   '
          f'U = {U.mean():.3f} ({U.std():.3f})')
    print(f'  {"r":>10} {"Theta(r)":>14} {"err":>10} {"counts":>10}')
    rows = [((r, 0, 0, 0), f'({r},0,0,0)') for r in range(1, args.N // 2 + 1)]
    rows.append(((args.N // 2,) * 4, 'antipode'))
    for x, name in rows:
        c = counts[x]
        if c > 0:
            print(f'  {name:>10} {mean[x]:>+14.6e} {err[x]:>10.1e} {c:>10.0f}')
        else:
            status = ('never visited: transport-censored'
                      if np.sqrt(sum(min(v, args.N - v)**2 for v in x)) > rmax
                      else 'no counts')
            print(f'  {name:>10} {"CENSORED":>14} {"":>10} {0:>10.0f}'
                  f'   Theta < {one_count:.1e} (naive)  [{status}]')
    return {x: (mean[x], err[x], counts[x]) for x, _ in rows}


# ---- shared thermalized valid start
rng = np.random.default_rng(args.seed)
g0 = gen.DefectGas(S, zeta=0.01, D_max=8, rng=rng)
phi = np.zeros((1,) + tuple(L.dims))
n = np.zeros((4,) + tuple(L.dims), dtype=np.int64)
phi, n = g0.run(phi, n, args.therm, tally=False)
phi, n = land_in_vacuum(phi, n, args.seed + 1)
print(f'thermalized valid start ready (N={args.N}, kappa={args.kappa})', flush=True)

# ---- tuned (campaign protocol)
z_tuned = gen.DefectGas.tune(S, D_max=8, rng=np.random.default_rng(args.seed + 2),
                             phi=phi, n=n, sweeps=150,
                             ladder=(0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005))
print(f'tuned zeta = {z_tuned:g} (dwell > 15% protocol)', flush=True)
e_A, gas_A, z_A = generate('tuned', z_tuned, 8, 4 * V, phi, n, args.seed + 3)
tab_A = report('tuned (campaign protocol)', e_A, gas_A, z_A, 4 * V)

# ---- edge
z_edge, emit_edge = gen.DefectGas.tune_edge(
    S, D_max=args.edge_D_max, rng=np.random.default_rng(args.seed + 4),
    phi=phi, n=n, floor=args.floor)
print(f'\nedge zeta = {z_edge:g}, emit_every = {emit_edge} (floor {args.floor})',
      flush=True)
e_B, gas_B, z_B = generate('edge', z_edge, args.edge_D_max, emit_edge, phi, n,
                           args.seed + 5)
tab_B = report(f'edge (floor {args.floor})', e_B, gas_B, z_B, emit_edge)

# ---- the two-zeta disagreement test where both runs have counts
print('\n=== two-zeta comparison (mean is zeta-independent; disagreement = '
      'finite-run transport bias)')
for x in tab_A:
    (mA, eA, cA), (mB, eB, cB) = tab_A[x], tab_B[x]
    if cA > 0 and cB > 0:
        sigma = abs(mA - mB) / np.sqrt(eA**2 + eB**2)
        print(f'  {str(x):>14}  tuned {mA:+.4e}({eA:.1e})  edge {mB:+.4e}({eB:.1e})'
              f'  -> {sigma:.2f} sigma')
    else:
        print(f'  {str(x):>14}  tuned counts {cA:.0f}, edge counts {cB:.0f} '
              f'-> no overlap')
