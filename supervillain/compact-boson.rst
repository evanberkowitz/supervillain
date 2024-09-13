
.. _compact-boson:

*****************
The Compact Boson
*****************

The compact boson in 1+1D

.. math ::

    \mathcal{Z} = \int \mathcal{D}\varphi\; \exp\left\{ - \int d^2x\; \frac{1}{8\pi} (\partial\varphi)^2\right\}

is dual to the free fermion and enjoys two interesting global symmetries,

.. math ::

    \begin{align}
        \text{shift}    && U(1)_S  &&  \varphi \rightarrow \varphi+\epsilon    &&  J^S_\mu &= \frac{i}{4\pi} \partial_\mu \varphi
        \\
        \text{winding}  && U(1)_W  &&  \text{topological, not Noetherian}      &&  J^V_\mu &= \frac{1}{2\pi} \epsilon_{\mu\nu} \partial^\nu \varphi
    \end{align}

which correspond to the vector and axial currents on the fermionic side.
The first is always conserved by the equations of motion of $\varphi$, but the second is only conserved so long as partial derivatives commute.
In other words, the winding symmetry is conserved *as long as there are no vortices*: parallel transport around vortices can yield a net winding number.

The modified Villain :cite:`Villain:1974ir` formulation of the compact boson is a lattice discretization which allows us to easily control the winding subgroup, allowing it to break completely (yielding the traditional XY model), forcing it to maintain a $\mathbb{Z}_W$ subgroup, or to keep it in its entirety.
This discretization and related physical models are implemented in supervillain.

The discretization is given by

.. math::

   \begin{align}
   Z[J] &= \sum\hspace{-1.33em}\int D\phi\; Dn\; Dv\; e^{-S[\phi, n, v]}
   &
   S[\phi, n, v] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2 + 2\pi i \sum_p v_p (dn)_p / W
   \end{align}

with $\phi\in\mathbb{R}$ on sites, $n\in\mathbb{Z}$ on links, and a Lagrange multiplier field $v\in\mathbb{Z}$ on plaquettes, and a careful choice of finite differencing $d$ that obeys $d^2=0$.
The path integral over $v$ restricts the vorticity plaquette-by-plaquette, setting $(dn) \equiv 0\; (\text{mod }W)$.

Changing the coupling $\kappa$ corresponds to dialing Thirring terms on the fermionic side; a particular value corresponds to the free fermion.
In general we do not have a simple map of the lattice coupling $\kappa$ to the radius of the compact boson; there may be special points where enhanced symmetries or self-duality protects $\kappa$ from renormalization.
