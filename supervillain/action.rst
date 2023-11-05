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
   n &\rightarrow\; n + dk

for an integer-valued 0-form $k$.

Computationally we can study this model in a variety of formulations.

.. autoclass :: supervillain.action.Villain
   :members:

.. _constrained:

=============================
The Constrained Villain Model
=============================

If we integrate over particular values of $J_p$ we can project out values of the winding $dn$.
For example, if we integrate $J$ over the reals the simplicity of the action allows us to find a constraint

.. math::
   \begin{align}
        \int DJ\; e^{i \sum_p J_p (dn)_p} = \prod_p 2\pi \delta(dn_p)
   \end{align}

but we may also set $J = 2\pi v / W$ for any positive integer $W$ and sum over integer-valued plaquette variables $v$,

.. math::
   \begin{align}
   Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; Dv\; e^{-S_J[\phi, n, v]}
   \\
   S_J[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p (v/W + J/2\pi) (dn)_p
   \end{align}

This model has a gauge symmetry $v \rightarrow v \pm W$ because with integer-valued $dn$ the phase

.. math::
    e^{2\pi i \sum_p v_p (dn)_p / W}

and the path integral are invariant under that transformation.

This model has a $\mathbb{Z}_W$ winding symmetry WHICH DESERVES A LOT MORE DISCUSSION HERE.

But for the unconstrained $W=1$, the obvious reading of this model has a horrible sign problem.
However, the sign problem can be traded for a constraint,

.. math::
        \sum Dv\; e^{2\pi i \sum_p v_p (dn)_p / W}
        =
        \prod_p 2\pi\; [dn_p \equiv 0 \text{ mod }W]

(where $[dn_p \equiv 0 \text{ mod } W]$ is the `Iverson bracket`_).
This constraint might be implemented with careful Monte Carlo updates.

Remarkably, we will see that the worldline formulation is naturally sign-problem free.

.. _Iverson bracket: https://en.wikipedia.org/wiki/Iverson_bracket

