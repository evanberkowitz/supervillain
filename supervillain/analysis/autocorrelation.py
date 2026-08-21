#!/usr/bin/env python

import numpy as np

from supervillain.batch import Batch
import supervillain

import logging
logger = logging.getLogger(__name__)

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

    The integrated autocorrelation time $\tau_{int}$ :cite:`Madras:1988ei` is

    .. math::
        \tau_{int} = \int_{0}^{\tau_0} d\tau\; C(\tau) = \frac{1}{2} + \sum_{\tau=1}^{\tau_0-1} C(\tau)

    where $\tau_0$ is the first time at which $C$ is no longer positive, and the sum stops before it.
    The $\frac{1}{2}$ is the trapezoidal weight of $C(0)=1$; the far endpoint carries no weight of its own, having fallen to zero.

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
    tau_0 = np.argmin(clamped)

    tau = np.ceil(0.5+C[1:tau_0].sum())
    return C, int(tau)


def autocorrelation_time(data, mean=None, weight=None):
    r'''
    Just like :func:`autocorrelation` but only returns $\tau_{int}$.  Pass
    ``weight`` to get the autocorrelation time of a :ref:`reweighted <reweighting>`
    estimator (computed on the influence function $w_t(O_t-\bar O)/\langle w\rangle$).
    '''
    _, tau = autocorrelation(data, mean, weight)
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

    # Half of nothing is nothing, and a τ of 0 is not a number this library can
    # mean: the minimum is 1, and it would be handed on as a Blocking width or an
    # every() stride, where it is nonsense a second time.  Refused here, before
    # any observable is attempted, so that an empty source does not first produce
    # a warning per observable from taking means of nothing.
    if len(source) == 0:
        raise ValueError(
            'there is no autocorrelation time; the source offers no samples at all.')

    # On a reweighted source the relevant tau is that of the ratio-estimator
    # influence function w(O-Obar)/<w>, not of O itself, so hand over the weights.
    # A no-op at unit weight, which is every source that carries no logWeight_
    # columns.  Because this is shared, a streamed or blocked source gets the same
    # treatment as an Ensemble.
    weight = Batch.as_array(source.weight)

    auto = dict()
    for name in observables:
        if not supervillain.observables[name].autocorrelation(source):
            continue
        try:
            auto[name] = autocorrelation_time(source.timeseries(name), weight=weight)
        except NotImplementedError:
            # The action has no such observable at all, which is not a diagnostic
            # and not the reader's business.  Only Villain and Worldline observables
            # are gated (by OnlyVillain and OnlyWorldline), so every other action
            # reaches here for most of the register, and reporting each one as a
            # failure to fluctuate would say something false about the ensemble.
            continue
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
