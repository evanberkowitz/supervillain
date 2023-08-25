#!/usr/bin/env python

import numpy as np

def autocorrelation(data, mean=None):
    r'''

    The *autocorrelation function* is
    
    .. math ::
        \begin{align}
        C(\tau) &= {\left\langle \Delta(t+\tau) \Delta(t) \right\rangle}
                   /    {\left\langle \Delta(t)^2              \right\rangle}
        &
        \Delta(t) &= \texttt{data}(t) - \texttt{mean}
        \end{align}

    where the ⟨averages⟩ are over the time $t$ and $C$ is normalized to 1 at $\tau=0$.

    The integrated autocorrelation time $\tau_{int}$ is

    .. math::
        \tau_{int} = \int_{0}^{\tau_0} d\tau\; C(\tau)

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

    Returns
    -------
    C: np.array
        The autocorrelation function $C$, the same length as the data.
    $\tau_{int}$: int
        The ceiling of the integrated autocorrelation time.
    '''
    if mean is None:
        mean = data.mean()

    Delta = data - mean

    plus = np.fft.fft(Delta, norm='backward')
    minus= np.fft.ifft(Delta, norm='forward')

    C = np.fft.fft(plus*minus, norm='backward').real / (len(Delta))**2
    C /= C[0] # normalize

    clamped = np.clip(C, 0, None)
    minIdx = np.argmin(clamped)
    return C, int(np.ceil(C[:minIdx].sum()))


def autocorrelation_time(data, mean=None):
    r'''
    Just like :func:`autocorrelation` but only returns $\tau_{int}$.
    '''
    _, tau = autocorrelation(data, mean)
    return tau

