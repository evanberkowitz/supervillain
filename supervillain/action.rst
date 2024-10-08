.. _action:

*****************
The Villain Model
*****************

We are interested in studying the Villain model with partition function $Z$ and action $S$ given by

.. math::
   :name: villain model

   \begin{align}
   Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S_J[\phi, n]}
   &
   S_J[\phi, n] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + i \sum_p J_p (dn)_p
   \end{align}

where $\phi$ is a real-valued 0-form that lives on sites $x$, $n$ is an integer-valued one-form that lives on links $\ell$, and $J$ is a two-form external source that lives on plaquettes $p$.
The model has a gauge symmetry

.. math::
   :label: gauge symmetry

   \phi &\rightarrow\; \phi + 2\pi k
   \\
   n &\rightarrow\; n + dk

for an integer-valued 0-form $k$.

If we integrate over particular values of $J_p$ we can project out values of the winding $dn$.
For example, if we integrate $J$ over the reals the simplicity of the action allows us to find a constraint

.. math::
   :name: vortex-free model

   \begin{align}
        \int DJ\; e^{i \sum_p J_p (dn)_p} = \prod_p 2\pi \delta(dn_p)
   \end{align}

which kills all vortices, because every plaquette must have 0 vorticity.
We may also set $J = 2\pi v / W$ for any positive integer $W$ and sum over integer-valued plaquette variables $v$,

.. math::
   :name: constrained villain model

   \begin{align}
   Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; Dv\; e^{-S_J[\phi, n, v]}
   \\
   S_J[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p (v/W + J/2\pi)_p (dn)_p,
   \end{align}

keeping the external $J$ for functional differentiation.
The constrained model has a gauge symmetry $v \rightarrow v \pm W$ because with integer-valued $dn$ the phase

.. math::
    e^{2\pi i \sum_p v_p (dn)_p / W}

and the path integral are invariant under that transformation.  When $W=1$ the ...constraint... does not constrain $dn$.
We may think of of the $U(1)_W$-maintaining :ref:`vortex-free model <vortex-free model>` as $W=\infty$.

The constrained model has a $\mathbb{Z}_W$ global winding symmetry

.. math ::

    \begin{align}
        v &\rightarrow v + z
        &
        (z&\in\mathbb{Z})
    \end{align}

which is harmless under the path integral of $v$ over the integers.

But for the unconstrained $W=1$, the obvious reading of this model has a horrible sign problem.
However, the sign problem can be traded for a constraint,

.. math::
   :name: winding constraint

        \sum Dv\; e^{2\pi i \sum_p v_p (dn)_p / W}
        =
        \prod_p [dn_p \equiv 0 \text{ mod }W]

(where $[dn_p \equiv 0 \text{ mod } W]$ is the `Iverson bracket`_).
This constraint might be implemented with careful Monte Carlo updates.
And we can sample configurations with $W=\infty$ if we can find an ergodic set of updates which never change $dn$ anywhere, assuming we start from a :ref:`vortex-free configuration <vortex-free model>`.

Remarkably, we will see that the worldline formulation is naturally sign-problem free.

Computationally we can study this model in a variety of formulations.  The most straightforwardly obvious is the literal one.

.. autoclass :: supervillain.action.Villain
   :members:

.. _Iverson bracket: https://en.wikipedia.org/wiki/Iverson_bracket

