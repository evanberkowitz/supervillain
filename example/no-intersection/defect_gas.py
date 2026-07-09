#!/usr/bin/env python

r"""
Command-line driver for :class:`~.DefectGas` --- the grand-canonical defect sampler
of the No-Intersection model, with the inline estimator of
$\langle e^{+i\theta_x} e^{-i\theta_y}\rangle$ (absolutely normalized).  See the class
docstring in ``supervillain/generator/no_intersection/fugacity.py`` for the algorithm
and dev/fugacity-worm.md for the design notes and validation.

Exactness self-test: the estimator is independent of --zeta (which tunes only the
variance); run twice with different values and compare.

Run from example/no-intersection/:

    uv run python defect_gas.py --N 6 --kappa 0.05 --tune --sweeps 2000
"""

import argparse

import numpy as np
from tqdm import tqdm

import supervillain
from supervillain.lattice import Lattice
import supervillain.generator.no_intersection as gen


parser = argparse.ArgumentParser(description='Grand-canonical defect worm.')
parser.add_argument('--N', type=int, default=6)
parser.add_argument('--kappa', type=float, default=0.05)
parser.add_argument('--zeta', type=float, default=0.5)
parser.add_argument('--D-max', type=int, default=8)
parser.add_argument('--sweeps', type=int, default=2000)
parser.add_argument('--thermalize', type=int, default=200,
                    help='untallied fugacity sweeps after the Hammer start')
parser.add_argument('--hammer', type=int, default=140,
                    help='constrained Hammer steps to thermalize the valid sector first')
parser.add_argument('--blocks', type=int, default=40)
parser.add_argument('--tune', action='store_true',
                    help='probe a zeta ladder and keep the first with healthy vacuum dwell')
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--out', default=None, help='npz path for the correlator')
args = parser.parse_args()

L = Lattice(4, args.N)
S = supervillain.action.NoIntersections(L, kappa=args.kappa)

# Thermalize INSIDE the valid sector with the constrained Hammer (fast, exact), then
# hand the hot valid configuration to the fugacity sampler.  Growing the dense low-κ
# sheet from cold is exactly what the constrained generators are good at; asking the
# fugacity moves to do it would waste sweeps paying ζ tolls for rearrangements the
# Hammer performs for free at q ≡ 0.
e = supervillain.Ensemble(S).generate(args.hammer, gen.Hammer(S), start='cold')
cfg = e.configuration[args.hammer - 1]
phi0, n0 = np.asarray(cfg['phi']), np.asarray(cfg['n'])

zeta = args.zeta
if args.tune:
    zeta = gen.DefectGas.tune(S, D_max=args.D_max,
                                 rng=np.random.default_rng(args.seed + 1),
                                 phi=phi0, n=n0)
    print(f'tuned zeta = {zeta}')

w = gen.DefectGas(S, zeta=zeta, D_max=args.D_max,
                     rng=np.random.default_rng(args.seed))
phi, n = w.run(phi0, n0, args.thermalize, tally=False, progress=tqdm)
w.blocks = []
w._new_block()
per_block = max(1, args.sweeps // args.blocks)
for b in tqdm(range(args.blocks)):
    phi, n = w.run(phi, n, per_block, tally=True)
    w.close_block()

G, dG = w.correlator()
U, dU = w.binder()
print(w.report())
print(f'ThetaBinderCumulant U = {U:.4f} ± {dU:.4f}   (Gaussian: 2 - 1/V; broken: -> 1)')
Lsym = Lattice(4, args.N)                     # throwaway for symmetrize caches
Gs, dGs = Lsym.symmetrize(G), Lsym.symmetrize(dG)
print(f'\n{"r":>3} {"Θ(r,0,0,0) [absolute; Θ(0) = 1 by definition]":>48}')
for r in range(1, args.N // 2 + 1):
    print(f'{r:>3} {Gs[(r,0,0,0)]:>+14.8f}±{dGs[(r,0,0,0)]:<12.8f}')
ap = (args.N // 2,) * 4
print(f'{"apd":>3} {Gs[ap]:>+14.8f}±{dGs[ap]:<12.8f}')
if args.out:
    np.savez(args.out, kappa=args.kappa, N=args.N, zeta=zeta,
             D_max=args.D_max, sweeps=args.sweeps, G=G, dG=dG,
             Gsym=Gs, dGsym=dGs, D_trace=np.array(w.D_trace),
             binder=U, dbinder=dU,
             H_four=sum(b[2] for b in w.blocks),
             H_Z=sum(b[1] for b in w.blocks))
    print(f'wrote {args.out}')
