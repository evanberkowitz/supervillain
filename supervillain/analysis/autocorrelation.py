#!/usr/bin/env python

import numpy as np

from supervillain.batch import Batch
import supervillain

import logging
logger = logging.getLogger(__name__)

def autocorrelation(data, mean=None, _cutoff=1e-16):
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

    The integrated autocorrelation time $\tau_{int}$ :cite:`Madras:1988ei` is

    .. math::
        \tau_{int} = \int_{0}^{\tau_0} d\tau\; C(\tau) = \frac{1}{2} + \sum_{\tau=1}^{\tau_0} C(\tau)

    where $\tau_0$ is the first time where $C$ is zero.

    .. note ::
        As defined, $t+\tau$ does not wrap around the end of the time series, because it makes no sense to say that the very end of a Markov chain influences the generation of the beginning.
        Nevertheless, this implementation does include correlations as though the Markov chain were periodic, to leverage Fourier acceleration of the convolution.

    Parameters
    ----------
    data: timeseries
        The data to correlate
    mean: float
        If `None`, compute the mean from the data.  But, if you know something about the quantity you're considering, you might want to impose a mean value rather than compute one from the data.
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

    if mean is None:
        mean = data.mean()

    Delta = data - mean

    plus = np.fft.fft(Delta, norm='backward')
    minus= np.fft.ifft(Delta, norm='forward')

    C = np.fft.fft(plus*minus, norm='backward').real / (len(Delta))**2
    if np.abs(C[0]) < _cutoff:
        raise ValueError('The fluctuations are too small to reliably determine an autocorrelation.')
    C /= C[0] # normalize

    clamped = np.clip(C, 0, None)
    tau_0 = np.argmin(clamped)

    tau = np.ceil(0.5+C[1:tau_0].sum())
    return C, int(tau)


def autocorrelation_time(data, mean=None):
    r'''
    Just like :func:`autocorrelation` but only returns $\tau_{int}$.
    '''
    _, tau = autocorrelation(data, mean)
    return tau



def sample_autocorrelation_time(source, observables=None, every=False):
    r'''
    The autocorrelation time of anything that can produce a timeseries.

    This is the shared implementation behind :meth:`.Ensemble.autocorrelation_time`
    and :meth:`.EnsembleStreamer.autocorrelation_time`; they differ only in where
    the timeseries comes from, not in which observables count or what to do when
    none of them fluctuate.

    ``source`` must provide ``Action``, ``__len__``, a ``measured`` collection of
    observable names, and ``timeseries(name)``.

    Parameters
    ----------
    observables: ``None`` or iterable of strings naming observables.
        Which observables to consider.  If ``None``, consider those already
        measured; if none have been, consider all of them.
    every: boolean
        If ``True`` returns a dictionary keyed by observable name.
    '''
    if observables is None:
        observables = set(o for o in source.measured
                          if supervillain.observables[o].autocorrelation(source))

    if len(observables) == 0:
        observables = tuple(supervillain.observables.keys())

    auto = dict()
    for name in observables:
        if not supervillain.observables[name].autocorrelation(source):
            continue
        try:
            auto[name] = autocorrelation_time(source.timeseries(name))
        except Exception:
            logger.warning(f'{name} does not fluctuate enough; it is not included in the autocorrelation time calculation.')

    if every:
        return auto

    if not auto:
        # Nothing fluctuated enough to estimate τ.  Rather than crash on an
        # empty max(), warn and fall back to half the length, which corresponds
        # to there being effectively a single independent sample
        # (N_eff = N / 2τ = 1).
        tau = int(np.ceil(len(source) / 2))
        logger.warning(
            'No observable fluctuated enough to estimate an autocorrelation time; '
            f'falling back to τ = {tau} (half the ensemble length).'
        )
        return tau

    return max(auto.values())
