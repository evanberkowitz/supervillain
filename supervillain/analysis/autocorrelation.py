#!/usr/bin/env python

import numpy as np

from supervillain.batch import Batch

def autocorrelation(data, mean=None, weight=None, _cutoff=1e-16):
    r'''

    The *autocorrelation function* is

    .. math ::
        \begin{aligned}
        C(\tau) &= {\left\langle \Delta(t+\tau) \Delta(t) \right\rangle}
                   /    {\left\langle \Delta(t)^2              \right\rangle}
        &
        \Delta(t) &= \texttt{data}(t) - \texttt{mean}
        \end{aligned}

    where the ⟨averages⟩ are over the time $t$ and $C$ is normalized to 1 at $\tau=0$.

    The integrated autocorrelation time $\tau_{int}$ is

    .. math::
        \tau_{int} = \int_{0}^{\tau_0} d\tau\; C(\tau)

    where $\tau_0$ is the first time where $C$ is zero.

    .. note ::
        As defined, $t+\tau$ does not wrap around the end of the time series, because it makes no sense to say that the very end of a Markov chain influences the generation of the beginning.
        Nevertheless, this implementation does include correlations as though the Markov chain were periodic, to leverage Fourier acceleration of the convolution.

    .. note ::
        On a :ref:`reweighted <reweighting>` ensemble the estimator is the ratio
        $\bar O = \langle wO\rangle/\langle w\rangle$, and the $\tau_{int}$ that
        inflates its variance is that of the influence function
        $f(t) = w_t\,(O_t - \bar O)/\langle w\rangle$, not of $O_t$.  Passing
        ``weight`` correlates $f(t)$ (which coincides with $O_t - \bar O$ when every
        weight is 1).  See :ref:`the analysis docs <weighted-autocorrelation>` for the derivation.

    Parameters
    ----------
    data: timeseries
        The data to correlate
    mean: float
        If `None`, compute the mean from the data (the ``weight``-weighted mean if
        ``weight`` is given).  But, if you know something about the quantity you're
        considering, you might want to impose a mean value rather than compute one.
    weight: timeseries or ``None``
        Per-configuration importance weights $w_t$ (e.g. :attr:`~.Ensemble.weight`).
        If given, the autocorrelation is computed on the ratio-estimator influence
        function $f(t) = w_t (O_t - \bar O)/\langle w\rangle$; if ``None`` the
        ordinary unweighted autocorrelation of $O_t - \texttt{mean}$ is used.
    _cutoff: float
        If $C(\tau=0)$ is less than the cutoff, there is a problem (for example, no fluctuations).

    Returns
    -------
    C: np.array
        The autocorrelation function $C$, the same length as the data.
    $\tau_{int}$: int
        The ceiling of the integrated autocorrelation time.
    '''
    data = Batch.as_array(data)

    if weight is None:
        if mean is None:
            mean = data.mean()
        Delta = data - mean
    else:
        # Weighted ensemble: correlate the ratio-estimator influence function
        # f(t) = w_t (O_t - Ō)/⟨w⟩ (zero-mean by construction), not O_t itself.
        w = Batch.as_array(weight)
        w = w.reshape((-1,) + (1,) * (data.ndim - 1))
        wbar = w.mean()
        Obar = (w * data).sum(axis=0) / w.sum(axis=0) if mean is None else mean
        Delta = w * (data - Obar) / wbar

    plus = np.fft.fft(Delta, norm='backward')
    minus= np.fft.ifft(Delta, norm='forward')

    C = np.fft.fft(plus*minus, norm='backward').real / (len(Delta))**2
    if np.abs(C[0]) < _cutoff:
        raise ValueError('The fluctuations are too small to reliably determine an autocorrelation.')
    C /= C[0] # normalize

    clamped = np.clip(C, 0, None)
    minIdx = np.argmin(clamped)
    return C, int(np.ceil(C[:minIdx].sum()))


def autocorrelation_time(data, mean=None, weight=None):
    r'''
    Just like :func:`autocorrelation` but only returns $\tau_{int}$.  Pass
    ``weight`` to get the autocorrelation time of a :ref:`reweighted <reweighting>`
    estimator (computed on the influence function $w_t(O_t-\bar O)/\langle w\rangle$).
    '''
    _, tau = autocorrelation(data, mean, weight)
    return tau

