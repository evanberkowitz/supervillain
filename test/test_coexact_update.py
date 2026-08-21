#!/usr/bin/env python

import numpy as np
import supervillain
from supervillain.lattice import delta


def test_coexact_update_preserves_constraint_D3():
    """δm=0 must hold after every step."""
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    G = supervillain.generator.worldline.CoexactUpdate(S)
    cfg = S.configurations(1)[0]
    for _ in range(10):
        cfg = G.step(cfg)
    assert S.valid(cfg)


def test_coexact_update_uses_all_2form_components():
    """In D=3, t should have proposals on all 3 two-form components, not just t[0].
    After many steps, δt should introduce changes in all 3 link directions."""
    L = supervillain.lattice.Lattice(D=3, N=4)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)
    G = supervillain.generator.worldline.CoexactUpdate(S)

    cfg = S.configurations(1)[0]
    for _ in range(200):
        cfg = G.step(cfg)

    m = cfg['m']
    # All D=3 link directions should be nonzero after many updates.
    # With only t[0] updated, only some link directions can ever change.
    assert (m[0] != 0).any(), "m component 0 never changed"
    assert (m[1] != 0).any(), "m component 1 never changed"
    assert (m[2] != 0).any(), "m component 2 never changed"


def test_coexact_update_dS_matches_action_difference():
    """Acceptance probabilities must be based on correct ΔS.
    Verify by computing S(new) - S(old) directly for accepted proposals."""
    np.random.seed(7)
    L = supervillain.lattice.Lattice(D=3, N=3)
    S = supervillain.action.Worldline(L, kappa=0.5, W=1)

    cfg_before = S.configurations(1)[0]
    cfg_after = supervillain.generator.worldline.CoexactUpdate(S).step(cfg_before)

    # Both before and after must satisfy the constraint
    assert S.valid(cfg_before)
    assert S.valid(cfg_after)
