#!/usr/bin/env python

r'''
Formatting of :class:`~supervillain.analysis.Uncertain`, with attention to the
degenerate cases an unattended campaign will eventually hand it.

An observable that vanishes identically is ordinary physics --- ``WindingSquared``
deep in the ordered phase, or ``TopologicalChargeDensitySquared`` in the
No-Intersection model, where the constraint sets it to zero configuration by
configuration.  Formatting such a value must not raise.
'''

import pytest

from supervillain.analysis import Uncertain


@pytest.mark.parametrize('uncertainty', (0., 1e-5, 0.25, 3.))
def test_zero_mean_formats(uncertainty):
    # log10(0) overflows; a zero mean has no scientific exponent of its own.
    formatted = f'{Uncertain(0., uncertainty)}'
    assert isinstance(formatted, str)
    assert formatted


def test_zero_mean_zero_uncertainty_is_zero():
    assert float(f'{Uncertain(0., 0.):}'.strip('+')) == 0.


def test_zero_mean_reports_the_uncertainty():
    # With no significant digits of the mean to be uncertain about, the +- form is
    # the honest presentation, and it must still carry the uncertainty.
    assert '0.25' in f'{Uncertain(0., 0.25)}'


@pytest.mark.parametrize('mean, uncertainty, expected', (
    (0.5, 0.003, '+5.000(30) × 10^-1'),
    (-0.5, 0.003, '-5.000(30) × 10^-1'),
    (8.5, 0.031, '+8.500(31)'),
))
def test_ordinary_values_are_unchanged(mean, uncertainty, expected):
    # The zero-mean guard must not perturb the normal path.
    assert f'{Uncertain(mean, uncertainty)}' == expected


@pytest.mark.parametrize('mean, uncertainty, expected', (
    (0.5, 0.003, '+5.000(30)e-1'),
    (-0.5, 0.003, '-5.000(30)e-1'),
))
def test_e_notation_is_unchanged(mean, uncertainty, expected):
    assert f'{Uncertain(mean, uncertainty):+eu2}' == expected
