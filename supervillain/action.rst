.. _action:

*****************
The Villain Model
*****************

We are interested in studying the Villain model with partition function $Z$ and action $S$ given by

.. math::
   \begin{align}
   Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S_J[\phi, n]}
   &
   S_J[\phi, n] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_p J_p (dn)_p
   \end{align}

where $\phi$ is a real-valued 0-form that lives on sites $x$, $n$ is an integer-valued one-form that lives on links $\ell$, and $J$ is a two-form external source that lives on plaquettes $p$.
The model has a gauge symmetry

.. math::
   \phi &\rightarrow\; \phi + 2\pi k
   \\
   n &\rightarrow\; n + 2\pi dk

for an integer-valued 0-form $k$.

Computationally we can study this model in a variety of formulations.

.. autoclass :: supervillain.action.Villain
   :members:

We can find an exact rewriting by first 'integrating' by parts,

.. math::
   \begin{align}
   S_J[\phi, n] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_\ell (\delta J)_\ell n_\ell
   \end{align}

and applying the Poisson summation formula

.. math::
   \sum_n \exp\left\{- \frac{\kappa}{2} (\theta - 2\pi n)^2 + i n \tilde{\theta}\right\}
   =
   \frac{1}{\sqrt{2\pi\kappa}} \sum_m \exp\left\{ - \frac{1}{2\kappa} \left(m - \frac{\tilde{\theta}}{2\pi}\right)^2 - i \left(m - \frac{\tilde{\theta}}{2\pi}\right) \theta\right\}

with $\theta \rightarrow\; d\phi$ and $\tilde{\theta} \rightarrow\; \delta J$ to find

.. math::
   \begin{align}
   Z[J] &=  (2\pi\kappa)^{-|\ell|/2}\sum\hspace{-1.33em}\int D\phi\; Dm\; e^{-S_J[\phi, m]}
   \\
   S_J[\phi, m] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 - i \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell (d\phi)_\ell.
   \end{align}

'Integrating' by parts again transforms the action to

.. math::
   S_J[\phi, m] = \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 - i \sum_x \left(\delta m - \frac{\delta^2 J}{2\pi}\right)_x \phi_x

and we may drop the $\delta^2 J$ term because $\delta^2=0$.
That leaves us with 

.. math::
   \begin{align}
   Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dm\; e^{-S_J[\phi, m]}
   &
   S_J[\phi, m] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 - i \sum_x \delta m _x \phi_x
   \end{align}

However, we can now execute the integral over $\phi$, which just sets $\delta m=0$ everywhere,

.. math::
   \begin{align}
   Z[J] &= \sum Dm\; e^{-S_J[m]} \left[\delta m = 0\right]
   &
   S_J[m] &= \frac{1}{2\kappa} \sum_\ell \left(m - \frac{\delta J}{2\pi}\right)_\ell^2 
   \end{align}

where $[\delta m = 0]$ is the `Iverson bracket`_.

.. autoclass :: supervillain.action.Worldline
   :members:


.. _Iverson bracket: https://en.wikipedia.org/wiki/Iverson_bracket

