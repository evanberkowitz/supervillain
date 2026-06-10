#!/usr/bin/env python

import numpy as np

from supervillain.analysis import Uncertain


def bootstrap_mean_stderr(series, *, draws=500, rng=None):
    r'''
    Bootstrap estimate of the mean and its standard error.

    Parameters
    ----------
    series: 1d array_like
        Per-draw samples (after thermalization / thinning).
    '''
    series = np.asarray(series)
    rng = np.random.default_rng(rng)
    n = len(series)
    resampled = rng.choice(series, (draws, n), replace=True)
    means = resampled.mean(axis=1)
    return means.mean(), means.std(ddof=1)


def compare_independent(mean_a, err_a, mean_b, err_b, *, sigma=3.0):
    r'''
    Compare two independent estimates.

    Returns
    -------
    dict with keys mean_a, mean_b, difference, combined_error, z, consistent
    '''

    diff = mean_a - mean_b
    combined = np.sqrt(err_a**2 + err_b**2)
    z = diff / combined if combined > 0 else 0.0
    return {
        'mean_a': mean_a,
        'mean_b': mean_b,
        'err_a': err_a,
        'err_b': err_b,
        'difference': diff,
        'combined_error': combined,
        'z': z,
        'consistent': abs(z) <= sigma,
    }


def _format_value(mean, err):
    if not np.isfinite(mean) or not np.isfinite(err):
        return f'{mean} ± {err}'
    if mean == 0 and err == 0:
        return '0'
    try:
        return f'{Uncertain(mean, err)}'
    except (OverflowError, ValueError):
        return f'{mean} ± {err}'


def format_comparison(name, result):
    flag = 'OK' if result['consistent'] else 'MISMATCH'
    return (
        f'{name:28s}  production {_format_value(result["mean_a"], result["err_a"])}  '
        f'compact {_format_value(result["mean_b"], result["err_b"])}  '
        f'Δ={_format_value(result["difference"], result["combined_error"])}  '
        f'z={result["z"]:+.2f}  [{flag}]'
    )
