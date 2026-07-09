:orphan:

.. _no_intersection_anomaly:

*****************************************************
Appendix: The Mixed 't Hooft Anomaly, Derived
*****************************************************

Let's demonstrate that the  :ref:`no-intersection model <no_intersection>` carries two exact $U(1)$ with a mixed 't Hooft anomaly of ABJ (axial--vector--vector) type :cite:`Adler1969,BellJackiw1969`.

Both symmetries act on the same object, so it is worth writing the path
integral out in full before gauging anything.  With $\phi$ a real 0-form on
sites, $n$ an integer 1-form on links, and $\theta$ a real 4-form on
hypercubes,

.. math ::
   :label: setup-action

   \begin{aligned}
       Z &= \sum\hspace{-1.33em}\int D\phi\; Dn\; D\theta\; e^{-S[\phi, n, \theta]}
       \\
       S[\phi, n, \theta] &= \frac{\kappa}{2} \sum_{\ell} (d\phi - 2\pi n)_\ell^2
       \;+\; i \sum_x \theta_x\, q_x, \qquad q = dn \wedge dn,
   \end{aligned}

reproducing :eq:`no-intersection` of the parent page. $\theta$ is a genuine
Lagrange multiplier here, not merely a label for the constraint: it is
integrated over the full real line at every hypercube, with no periodicity
imposed by hand. Carrying out that integral first is what turns
:eq:`setup-action` into the constrained partition function actually sampled
by the Monte Carlo,

.. math ::

   \int D\theta\; e^{i\sum_x \theta_x q_x} = \prod_x 2\pi\,\delta(q_x)
   \qquad\Longrightarrow\qquad
   Z = \sum\hspace{-1.33em}\int D\phi\; Dn\; e^{-S_{\text{Villain}}[\phi,n]} \prod_x [q_x = 0],

exactly as $J$ is integrated out to impose $dn = 0$ in the ordinary modified Villain
construction :ref:`vortex-free model <vortex-free model>`. For the anomaly, though, $\theta$ must stay
explicit: its shift symmetry is only visible in the *unintegrated* presentation
:eq:`setup-action`.  Everything
below works with $\theta$ present and never performs that integral.

The derivation uses the standard diagnostic for a mixed anomaly: gauge one
symmetry with a background field, and check whether the *other* symmetry
survives.  If it doesn't --- if the partition function picks up a
background-dependent phase under the second symmetry's transformation that no
local counterterm can remove --- the two symmetries have a mixed anomaly
:cite:`tHooft1980`.  Every step below is an exact lattice identity, in the
same sense that $Q = \sum_x q_x = 0$ is exact even at finite lattice spacing: nothing here
relies on a continuum limit or a smoothness assumption.  That kind of
exactness is the point of the modified-Villain lattice program more broadly
--- gauging a global symmetry by minimally coupling to a background, and
reading off the anomaly from a lattice identity for the coupled topological
density, is exactly the technique used to realize continuum anomalies exactly
at finite lattice spacing in two dimensions
:cite:`GattringerSulejmanpasic2019,Berkowitz:2023pnz` and to build an exact
lattice $\theta$-term in four dimensions via the Pontryagin square
:cite:`JacobsonSulejmanpasic2023`; the momentum--winding mixed anomaly of the
2D compact boson is the same statement one form degree down
:cite:`GorantlaLamSeibergShao2021,PaceChatterjeeShao2025`.  This particular
4D application --- vortex-sheet self-intersection as the "instanton number"
whose conjugate angle is $\theta$ --- is Jacobson's construction and is not
yet published elsewhere, but the technique for extracting its anomaly is
standard.

The Two Global Symmetries
=========================

Recall the field content: $\phi$ a real 0-form on sites, $n$ an integer
1-form on links, and $\theta$ a real, non-dynamical 4-form on hypercubes
multiplying the topological charge density $q = dn \wedge dn$
:eq:`setup-action`.

.. math ::
   :label: two-u1s

   \begin{aligned}
       U(1)_\phi&:&\phi &\rightarrow\; \phi + \alpha & \alpha \in&\; \mathbb{R}/2\pi\mathbb{Z},
       \\
       U(1)_\theta&:&\theta &\rightarrow\; \theta + \beta & \beta \in&\; \mathbb{R}
   \end{aligned}

with $\alpha$ and $\beta$ global constants.
The $U(1)_\phi$ is a symmetry of the Villain term $(d\phi - 2\pi n)^2$ for any
constant $\alpha$, since $d\alpha = 0$; this is the "standard Villain $U(1)$"
present in every model of this family, with Noether current $j = \kappa(d\phi
- 2\pi n)$ and on-shell conservation $\delta j = 0$.

$U(1)_\theta$ is a symmetry of the constraint term for any constant $\beta$
because the total charge vanishes *identically*,

.. math ::

   \sum_x q_x \;=\; \sum_x (dn \wedge dn)_x \;=\; \sum_x d(n \wedge dn)_x \;=\; 0 ,

a discrete-Stokes identity: the sum of an exact 4-form over every hypercube of a closed
lattice vanishes.  Because this holds configuration by configuration --- not
just on average --- shifting $\theta$ by *any* real $\beta$, not merely a
multiple of $2\pi$, leaves $e^{i\sum_x \theta_x q_x}$ exactly invariant

.. math ::

   e^{i\sum_x \theta_x q_x} \;\longrightarrow\; e^{i\sum_x (\theta_x + \beta) q_x} \;=\; e^{i\beta \sum_x q_x + i\sum_x \theta_x q_x} \;=\; e^{i\sum_x \theta_x q_x}.

What makes the mixed 't Hooft anomaly exciting is that neither symmetry has its own anomaly.
Both symmetries above are unconditional: $U(1)_\phi$'s invariance never asked
anything of $n$ or $\theta$, and $U(1)_\theta$'s invariance never asked
anything of $\phi$.  Each symmetry, gauged by itself, is anomaly-free ---
there is no obstruction to coupling either one individually to its own
background field. The anomaly, if any, can only show up in the cross term,
which is exactly where a *mixed* anomaly must live.

Gauging $U(1)_\phi$: the covariant charge density
===================================================

Introduce a background connection $A$ for $U(1)_\phi$ --- a real 1-form on
links, with the same quantization as $2\pi n$, so that closed 2-surfaces
carry flux $\oint dA \in 2\pi\mathbb{Z}$ --- and minimally couple:

.. math ::

   (d\phi - 2\pi n) \;\longrightarrow\; (d\phi - 2\pi n - A).

This is inert under the lattice gauge symmetry $\phi \to \phi + 2\pi k,\ n \to
n + dk$ and under background gauge transformations $\phi \to \phi + \lambda,\
A \to A + d\lambda$ for any real $\lambda$, exactly as the ungauged
combination was inert under the first alone.

The charge density $q$ must be covariantized the same way. This is the same
substitution used to build a background-covariant instanton/self-intersection
density in the modified-Villain literature cited above: the flux that enters
the topological term is not $n$'s flux alone but its difference from the
background's own would-be flux,

.. math ::
   :label: covariant-charge

   dn \;\longrightarrow\; dn - \frac{dA}{2\pi}, \qquad
   q^A \;\equiv\; \left(dn - \frac{dA}{2\pi}\right) \wedge \left(dn - \frac{dA}{2\pi}\right).

$q^A$ is manifestly invariant under $A \to A + d\lambda$, since $d(d\lambda) =
0$, so it depends on $A$ only through the physical background curvature $F =
dA$.

The anomalous total charge
===========================

Expand the covariant charge density :eq:`covariant-charge` and sum over every hypercube of the closed
lattice.  Despite the lattice wedge's :ref:`failure to have anti/commutativity <lattice-wedge-commutativity-fail>` we can simplify dramatically using bilinearity and the Leibniz rule :eq:`leibniz-rule`.

.. math ::
   :label: q-expansion

   \begin{aligned}
       Q^A \;\equiv\; \sum_x q^A_x
       =& \sum_x (dn \wedge dn)_x
       \nonumber\\
       &- \frac{1}{2\pi} \sum_x \big[(dn \wedge dA)_x + (dA \wedge dn)_x\big]
       \nonumber\\
       &+ \frac{1}{4\pi^2} \sum_x (dA \wedge dA)_x .
   \end{aligned}

The first three lines vanish for the same reason: they are all exact 4-forms,

.. math::

   \begin{aligned}
       da \wedge db &= d(a \wedge db)
       &
       \sum_x (da \wedge db)_x = \sum_x d(a \wedge db)_x = 0
   \end{aligned}

on a closed lattice.

The last term simplifies to the second Chern number of the background,

.. math ::
   :label: three-terms

   \begin{aligned}
      \frac{1}{4\pi^2}\sum_x (dA \wedge dA)_x &= \frac{1}{4\pi^2}\int F \wedge F \;\equiv\; k[A] \in \mathbb{Z}
   \end{aligned}

So

.. math ::
   :label: anomalous-charge

   Q^A \;=\; k[A] \in \mathbb{Z}, \qquad \text{generically } k[A] \neq 0.

The total charge, which was forced to vanish identically at $A = 0$, is now
pinned to an integer characteristic class of the background instead.
The result $Q^A = k[A]$ counts how many times the background
$U(1)_\phi$ bundle self-links, in exactly the sense that $\oint dA \in
2\pi\mathbb{Z}$ makes $F$ the curvature of a consistent connection. On the
periodic lattice a convenient way to make $k[A]$ nonzero is to thread
independent flux through two disjoint 2-cycles, $\oint_{01} F = 2\pi m_{01}$
and $\oint_{23} F = 2\pi m_{23}$, giving $k[A]$ proportional to the product
$m_{01} m_{23}$ and nonzero for generic integer choices.

The Mixed 't Hooft Anomaly
==========================

Now transform by $U(1)_\theta$ in the presence of the background:

.. math ::

   e^{i\sum_x \theta_x q^A_x} \;\longrightarrow\; e^{i\beta Q^A}\, e^{i\sum_x \theta_x q^A_x}
   \;=\; e^{i\beta\, k[A]}\, e^{i\sum_x \theta_x q^A_x}.

At $A = 0$ this phase is trivial for every real $\beta$, which is exactly the
exact $U(1)_\theta$ found above. Once $U(1)_\phi$ is gauged with a
topologically nontrivial background, $k[A]$ is a nonzero integer for generic
flux, and the phase $e^{i\beta k[A]}$ is nontrivial for continuous $\beta$:
the continuous shift symmetry is explicitly broken by the background. Only
its $2\pi\mathbb{Z}$ subgroup survives, since $e^{i 2\pi k[A]} = 1$ for any
integer $k[A]$ --- exactly the periodicity that must survive, because $q$
(and hence $q^A$) is integer-valued pointwise regardless of the background.
No local counterterm built from $A$ alone can repair this: the anomalous charge :eq:`anomalous-charge`
is a bulk identity fixed by the topology of $A$, not a boundary artifact, so
there is no choice of local counterterm added to the action that cancels
$k[A]$ while remaining a functional of $A$ alone. By the symmetric argument
--- gauge $U(1)_\theta$ instead and ask what happens to $U(1)_\phi$ --- the
same obstruction reappears with the roles exchanged. Neither $U(1)$ can be
gauged without an explicit background-dependent violation of the other. That
obstruction, present only when *both* backgrounds are turned on, is precisely
a mixed 't Hooft anomaly :cite:`tHooft1980`.

Matching the ABJ (AVV) structure
=================================

The anomalous phase $e^{i\beta k[A]} = e^{i\beta \int F\wedge F / 4\pi^2}$ is
literally the ABJ triangle in the language of forms
:cite:`Adler1969,BellJackiw1969`: $U(1)_\theta$ plays the role of the axial
current, whose would-be continuous conservation is spoiled by an anomalous
term proportional to $F \wedge F$, and $U(1)_\phi$ plays the role of the
vector current appearing *twice* --- both legs of the anomaly diagram are the
same background field $A$, which is exactly what "axial--vector--vector"
means: one axial insertion, two identical vector insertions.

$k[A]$ itself, though, is not quite the number usually quoted in the ABJ
story, and it is worth being precise about the difference rather than
sweeping it under the analogy. The continuum ABJ anomaly counts fermion zero
modes through the Atiyah--Singer index of the Dirac operator coupled to $A$,
which on a flat manifold is $\text{ind} = \int F\wedge F / 8\pi^2 = k[A]/2$
--- *half* of the self-intersection number derived above. That factor of two
is not a discrepancy to fix; it tracks exactly the fact that $\text{ind}$
requires a spin structure (for the fermions to exist at all) while $k[A] =
c_1^2[F]$ does not, being an integer for any background on any closed
oriented 4-manifold by unimodularity of the cup product on $H^2(X;\mathbb{Z})$
alone. On the 4-torus, which is spin, the intersection form is *even*, so
$c_1^2[F]$ --- hence $k[A]$ --- is guaranteed even, and $k[A]/2 = \text{ind}$
lands on an integer as it must. The two conventions are consistent, not
competing: $k[A]$ is the purely bosonic quantity our lattice derivation
actually produces, with no fermions or spin structure invoked anywhere in
:eq:`covariant-charge` through :eq:`anomalous-charge`, and it happens to carry
exactly twice the information of the fermionic index for the same topological
reason the index is well defined at all.

None of this touches the anomaly itself: :eq:`anomalous-charge` only needs
$k[A]$ to be a nonzero integer for generic background flux, which it is in
either convention, and Fujikawa's calculation of the anomalous measure
Jacobian is what ties an axial rotation to a shift of the $\theta$-angle by
that same integer in the fermionic story. Here there are no fermions and no
continuum regularization to worry about: :eq:`anomalous-charge` is the exact
lattice replacement for that index-theorem statement, valid at any $\kappa$
and any $N$. That finite-lattice-spacing exactness is the entire point of
building the constraint out of forms that satisfy the Leibniz rule and $d^2 =
0$ on the nose, the same discrete machinery already used for $q = dJ$ in the
parent page and for the sibling 2D chiral anomaly :cite:`Berkowitz:2023pnz`
and 4D Pontryagin-square $\theta$-term :cite:`JacobsonSulejmanpasic2023`
constructions cited above.

Consequences for the phase diagram
====================================

A mixed 't Hooft anomaly must be matched at every scale
:cite:`tHooft1980`: whatever the long-distance description of the
no-intersection model turns out to be, it has to reproduce the same
$U(1)_\phi$--$U(1)_\theta$ AVV anomaly computed here at the lattice level. In
particular, a trivially gapped phase --- a unique, symmetric vacuum with no
massless modes and no topological order --- cannot match a mixed anomaly, so
no $\kappa$ can give that outcome while leaving *both* symmetries unbroken.
That leaves exactly the menu the parent page already lays out on physical
grounds: spontaneous breaking of one $U(1)$ at a time (with a transition, of
some order, between the two broken phases), or a single anomaly-matching
gapless fixed point where both symmetries survive together as they do in the
handful of known 4D CFTs with fermionic AVV anomalies. The anomaly derived
here is what makes that last option --- and the 4D bosonization question it
raises --- more than idle speculation: it is the one algebraic fact the
putative CFT would be obligated to reproduce.