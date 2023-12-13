.. _worldline:

*************************
The Worldline Formulation
*************************

We can find an exact rewriting of the constrained :class:`~.Villain` model by first 'integrating' by parts,

.. math::
   \begin{align}
   S_J[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_\ell \delta(v/W + J/2\pi)_\ell n_\ell
   \end{align}

and applying link-by-link the Poisson summation formula

.. math::
   \sum_n \exp\left\{- \frac{\kappa}{2} (\theta - 2\pi n)^2 + i n \tilde{\theta}\right\}
   =
   \frac{1}{\sqrt{2\pi\kappa}} \sum_m \exp\left\{ - \frac{1}{2\kappa} \left(m - \frac{\tilde{\theta}}{2\pi}\right)^2 - i \left(m - \frac{\tilde{\theta}}{2\pi}\right) \theta\right\}

with $\theta \rightarrow\; d\phi$ and $\tilde{\theta} \rightarrow\; \delta (2\pi v / W + J)$ to find

.. math::
   \begin{align}
   Z[J] &=  (2\pi\kappa)^{-|\ell|/2}\sum\hspace{-1.33em}\int D\phi\; Dm\; Dv\; e^{-S_J[\phi, m, v]}
   \\
   S_J[\phi, m, v] &= \frac{1}{2\kappa} \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi}\right)\right)_\ell^2 - i \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi}\right)\right)_\ell (d\phi)_\ell.
   \end{align}

'Integrating' by parts again transforms the action to

.. math::
   S_J[\phi, m, v] = \frac{1}{2\kappa} \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi} \right)\right)_\ell^2 - i \sum_x \left(\delta m\right)_x \phi_x

and we dropped the $\delta^2$ term because $\delta^2=0$.
That leaves us with 

.. math::
   \begin{align}
   Z[J] &= (2\pi\kappa)^{-|\ell|/2} \sum\hspace{-1.33em}\int D\phi\; Dm\; Dv\; e^{-S_J[\phi, m, v]}
   \\
   S_J[\phi, m, v] &= \frac{1}{2\kappa} \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi}\right)\right)_\ell^2 - i \sum_x (\delta m)_x \phi_x
   \end{align}

However, we can now execute the integral over $\phi$, which just sets $\delta m=0$ everywhere,

.. math::
   \begin{align}
   Z[J] &= (2\pi)^{|x|}(2\pi\kappa)^{-|\ell|/2} \sum Dm\; Dv\; e^{-S_J[m, v]} \left[\delta m = 0\right]
   \\
   S_J[m, v] &= \frac{1}{2\kappa} \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi} \right)\right)_\ell^2 
   \end{align}

where $[\delta m = 0]$ is the `Iverson bracket`_ and we picked up a $2\pi$ for every site since $\int d\phi\; e^{i o \phi} = 2\pi \delta(o)$.
We can cast the dimensionless constants up into the action

.. math::
   \begin{align}
   Z[J] &= \sum Dm\; Dv\; e^{-S_J[m, v]} \left[\delta m = 0\right]
   \\
   S_J[m, v] &= \frac{1}{2\kappa} \sum_\ell \left(m - \delta\left(\frac{v}{W} + \frac{J}{2\pi} \right)\right)_\ell^2 + \frac{|\ell|}{2} \ln (2\pi \kappa) - |x| \ln 2\pi
   \end{align}

which will make functional differentiation more straightforward.

.. autoclass :: supervillain.action.Worldline
   :members:

.. _Iverson bracket: https://en.wikipedia.org/wiki/Iverson_bracket

