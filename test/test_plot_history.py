#!/usr/bin/env python

import pytest
import supervillain

# plot_history needs matplotlib; skip cleanly where it is not installed.
plt = pytest.importorskip('matplotlib.pyplot')
import matplotlib
matplotlib.use('Agg')


def _ensemble():
    S = supervillain.action.Villain(supervillain.lattice.Lattice(D=2, N=4), kappa=0.5, W=1)
    return supervillain.Ensemble(S).generate(
        30, supervillain.generator.villain.Hammer(S), start='cold')


def test_plot_history_labels_do_not_bleed_across_calls():
    # Regression: the default history_kwargs must not be a shared mutable dict that carries the
    # first call's label into the second.
    e1, e2 = _ensemble(), _ensemble()
    fig, ax = plt.subplots(1, 2)
    e1.plot_history(ax, 'ActionDensity', label='first')
    e2.plot_history(ax, 'ActionDensity', label='second')
    assert [line.get_label() for line in ax[0].get_lines()] == ['first', 'second']
    plt.close(fig)


def test_plot_history_does_not_mutate_callers_kwargs():
    # The caller's own history_kwargs dict must be left untouched (no in-place 'label' insert).
    e = _ensemble()
    fig, ax = plt.subplots(1, 2)
    mine = {}
    e.plot_history(ax, 'ActionDensity', label='x', history_kwargs=mine)
    assert mine == {}
    plt.close(fig)
