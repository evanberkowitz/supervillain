#!/usr/bin/env python

import numpy as np


def link_activity(ensemble, field='n'):
    r"""
    The mean absolute change of a field between consecutive configurations, per link:

    .. math::
        a_\ell = \frac{1}{|E|-1} \sum_{t} \left| f_\ell(t+1) - f_\ell(t) \right|

    A Monte Carlo stream that is healthy everywhere has $a_\ell > 0$ on every link;
    a *frozen region* — a locally blocked texture that no local update can touch —
    announces itself as a persistent spatial hole of zeros.  Because it only compares
    stored configurations, this diagnostic sees the combined effect of **all**
    generators in the stream without instrumenting any of them.

    Parameters
    ----------
    ensemble:
        A :class:`~supervillain.Ensemble` (its ``configuration`` field is used) or an
        iterable of configuration dicts or of plain field arrays.
    field: str
        Which field to difference when configurations are dicts (default ``'n'``).

    Returns
    -------
    np.ndarray
        The per-link (or per-component) mean |Δfield|, with the shape of one field.
    """
    if hasattr(ensemble, 'configuration'):
        ensemble = ensemble.configuration
    fields = [np.asarray(c[field]) if isinstance(c, dict) else np.asarray(c)
              for c in ensemble]
    if len(fields) < 2:
        raise ValueError('link_activity needs at least two configurations to difference.')
    return np.abs(np.diff(np.stack(fields), axis=0)).mean(axis=0)


def site_activity(ensemble, field='n'):
    r"""
    :func:`link_activity` reduced to sites by summing the component axis: the mean
    total |Δfield| based at each site.  Convenient for spotting spatial holes.

    Returns
    -------
    np.ndarray
        Shape ``(N, ..., N)``.
    """
    return link_activity(ensemble, field=field).sum(axis=0)
