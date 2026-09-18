import numpy as np
from itertools import product, permutations
from math import factorial

import supervillain.lattice.barycentric as b


def test_type_counts_match_closed_form():
    """T_p(D) = sum_j (-1)^j C(p,j) (2(p-j)+1)^D."""
    from math import comb
    for D in (2, 3, 4):
        s = b._structures(D)
        for p in range(D + 1):
            expected = sum((-1)**j * comb(p, j) * (2*(p-j)+1)**D
                           for j in range(p + 1))
            assert s.type_count[p] == expected, f'D={D} p={p}'


def test_four_simplex_count_is_384():
    """Note eq. (3): 16 * 4! = 384 fine four-simplices per hypercube."""
    assert b._structures(4).type_count[4] == 384


def test_every_top_type_is_a_maximal_flag():
    """A maximal flag adds one dimension at a time, so every step is a single direction."""
    s = b._structures(4)
    for steps in s.step_types[4]:
        assert all(len(step) == 1 for step in steps)


def test_offsets_are_cumulative_signed_unit_steps():
    s = b._structures(3)
    for t, steps in enumerate(s.step_types[3]):
        v = np.zeros(3, dtype=int)
        assert (s.offsets[3][t][0] == v).all()
        for i, step in enumerate(steps):
            for mu, sign in step:
                v[mu] += sign
            assert (s.offsets[3][t][i + 1] == v).all()


def test_measure_of_every_top_simplex_is_one_over_384():
    """Note section 2.1: each fine four-simplex has volume 1/384."""
    s = b._structures(4)
    assert np.allclose(s.measure[4], 1.0 / 384)


def test_measure_of_top_simplex_is_general_formula():
    for D in (2, 3, 4):
        s = b._structures(D)
        assert np.allclose(s.measure[D], 1.0 / (factorial(D) * 2**D))


def test_link_length_is_sqrt_rank_difference_over_two():
    """Note section 2.1: a class (r,s) link has length sqrt(s-r)/2."""
    s = b._structures(4)
    for t, steps in enumerate(s.step_types[1]):
        k = len(steps[0])                      # directions released = s - r
        assert np.isclose(s.measure[1][t], np.sqrt(k) / 2)


def test_blue_triangle_has_area_one_eighth():
    """Note section 2.4: a plaquette is cut into eight fine triangles."""
    s = b._structures(4)
    blue = s.type_index[2][(((0, +1),), ((1, +1),))]
    assert np.isclose(s.measure[2][blue], 1.0 / 8)


def test_orientation_sign_matches_determinant():
    """eps_sigma is the sign of det[v_k - v_0]."""
    for D in (2, 3, 4):
        s = b._structures(D)
        for t in range(s.type_count[D]):
            W = s.offsets[D][t][1:]            # (D, D), doubled units
            assert s.orientation_sign[t] == round(np.sign(np.linalg.det(W)))


def test_orientation_sign_closed_form():
    """For a maximal flag, eps = (prod of signs) * (sign of the direction order)."""
    s = b._structures(4)
    for t, steps in enumerate(s.step_types[4]):
        dirs  = [step[0][0] for step in steps]
        signs = [step[0][1] for step in steps]
        parity = sum(1 for i in range(4) for j in range(i + 1, 4)
                     if dirs[i] > dirs[j]) % 2
        assert s.orientation_sign[t] == np.prod(signs) * (-1)**parity


def test_ten_link_classes():
    """Note section 2.5: C(5,2) = 10 classes in D = 4."""
    from math import comb
    for D in (2, 3, 4):
        s = b._structures(D)
        assert len(s.link_classes) == comb(D + 1, 2)
    assert b._structures(4).link_classes == (
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4), (3, 4))


def test_site_rank_counts_odd_coordinates():
    lat = b.Lattice(3, 3)
    assert lat.site_rank.shape == (6, 6, 6)
    assert lat.site_rank[0, 0, 0] == 0
    assert lat.site_rank[1, 0, 0] == 1
    assert lat.site_rank[1, 1, 1] == 3


def test_sixteen_fine_vertices_per_original_site():
    """A closed hypercube has 16+32+24+8+1 = 81 barycenters; after periodic
    identification the complex has 2**D = 16 fine vertices per original site."""
    lat = b.Lattice(4, 3)
    assert lat.site_rank.size == (2 * 3)**4
    assert lat.site_rank.size / 3**4 == 16
    counts = np.bincount(lat.site_rank.ravel())
    from math import comb
    assert list(counts) == [comb(4, r) * 3**4 for r in range(5)]


def test_two_hundred_forty_links_per_original_site():
    """Note section 2.5, with the ten multiplicities."""
    lat = b.Lattice(4, 3)
    assert lat.count(1) / 3**4 == 240
    tally = np.bincount(lat.link_class[lat.link_class >= 0].ravel())
    got = {lat.link_classes[i]: n / 3**4 for i, n in enumerate(tally)}
    assert got == {(0, 1): 8, (0, 2): 24, (1, 2): 24,
                   (2, 3): 24, (2, 4): 24, (3, 4): 8,
                   (0, 3): 32, (0, 4): 16, (1, 3): 48, (1, 4): 32}


def test_total_measure_of_top_simplices_is_the_volume():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = b.Lattice(D, N)
        assert np.isclose((lat.measure(D) * lat.support(D)).sum(), N**D)


def test_form_carries_its_degree():
    lat = b.Lattice(3, 3)
    a = lat.zeros(2)
    assert isinstance(a, b.Form)
    assert a.degree == 2
    assert a.lattice is lat
    assert a.shape == (b._structures(3).type_count[2],) + (6, 6, 6)


def test_same_degree_arithmetic_stays_a_form():
    lat = b.Lattice(3, 3)
    a, c = lat.random(2), lat.random(2)
    assert isinstance(a + c, b.Form) and (a + c).degree == 2
    assert isinstance(2 * a, b.Form) and (2 * a).degree == 2
    assert isinstance(np.isclose(a, c), b.Form)


def test_reductions_degrade_to_plain_numpy():
    lat = b.Lattice(2, 3)
    a = lat.random(1)
    assert not isinstance(a.sum(), b.Form)
    assert not isinstance(a.max(), b.Form)


def test_mixed_degree_arithmetic_degrades_to_ndarray():
    """Two Forms of different degree have no unambiguous result degree, so the
    product is a plain ndarray -- the same rule compact.Form follows."""
    lat = b.Lattice(2, 3)
    a, c = lat.random(0), lat.random(0)
    # Degree 0 has T_0 = 1 while degree 1 has T_1 = 8, so broadcast a matching
    # pair by hand: take one type slice of each and re-wrap at different degrees.
    one = b.Form(np.asarray(lat.random(1))[:1], degree=1, lattice=lat)
    two = b.Form(np.asarray(a)[:1],             degree=0, lattice=lat)
    assert not isinstance(one * two, b.Form)


def test_random_respects_support():
    lat = b.Lattice(3, 3)
    for p in range(4):
        a = lat.random(p)
        assert (np.asarray(a)[lat.support(p) == 0] == 0).all()


def test_accessors_are_memoized():
    lat = b.Lattice(3, 3)
    assert lat.support(2) is lat.support(2)
    assert lat.measure(2) is lat.measure(2)


def test_push_rolls_whole_unit_cells():
    """A unit translation must move a site by 2 in doubled coordinates."""
    lat = b.Lattice(3, 3)
    a = lat.random(1)
    assert np.allclose(np.asarray(b.push(a, (1, 0, 0))),
                       np.roll(np.asarray(a), 2, axis=1))


def test_push_preserves_support_and_degree():
    lat = b.Lattice(3, 3)
    for p in range(4):
        a = lat.random(p)
        moved = b.push(a, (1, 2, 0))
        assert moved.degree == p
        assert (np.asarray(moved)[lat.support(p) == 0] == 0).all()


def test_pull_inverts_push():
    lat = b.Lattice(3, 3)
    a = lat.random(2)
    assert np.allclose(np.asarray(b.pull(b.push(a, (1, 2, 1)), (1, 2, 1))),
                       np.asarray(a))


def test_translating_by_N_is_the_identity():
    lat = b.Lattice(3, 3)
    a = lat.random(1)
    assert np.allclose(np.asarray(b.translate(a, (3, 3, 3))), np.asarray(a))


def test_gather_is_a_half_step():
    lat = b.Lattice(2, 3)
    raw = np.arange((2 * 3)**2, dtype=float).reshape(6, 6)
    assert np.allclose(b._gather(raw, (1, 0)), np.roll(raw, -1, axis=0))


def test_d_raises_the_degree():
    lat = b.Lattice(3, 3)
    a = lat.random(1)
    assert b.d(a).degree == 2


def test_d_squared_is_zero():
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = b.Lattice(D, N)
        for p in range(D - 1):
            a = lat.random(p)
            da = b.d(a)
            assert not np.allclose(np.asarray(da), 0), f'D={D} p={p} da vanished'
            assert np.allclose(np.asarray(b.d(da)), 0), f'D={D} p={p}'


def test_d_output_lives_on_real_simplices():
    lat = b.Lattice(3, 3)
    da = b.d(lat.random(1))
    assert (np.asarray(da)[lat.support(2) == 0] == 0).all()


def test_flux_is_closed():
    """f = dn is exact, hence df = 0 identically."""
    lat = b.Lattice(4, 3)
    n = lat.random_integer(1)
    f = b.d(n)
    assert not np.allclose(np.asarray(f), 0), (
        'the flux f = dn vanished identically -- df = 0 would then hold '
        'vacuously.')
    assert np.allclose(np.asarray(b.d(f)), 0)


def test_d_on_a_zero_cochain_is_the_difference_of_endpoints():
    """Note eq. (4) at p = 0: (d phi)_01 = phi_1 - phi_0."""
    lat = b.Lattice(2, 3)
    s = b._structures(2)
    phi = lat.random(0)
    dphi = b.d(phi)
    t = s.type_index[1][(((0, +1),),)]          # the link from xi to xi + e_0
    raw = np.asarray(phi)[0]
    assert np.allclose(np.asarray(dphi)[t] * lat.support(1)[t],
                       (np.roll(raw, -1, axis=0) - raw) * lat.support(1)[t])


def test_delta_lowers_the_degree():
    lat = b.Lattice(3, 3)
    assert b.delta(lat.random(2)).degree == 1


def test_adjointness():
    """sum (da).b = sum a.(delta b) -- the strongest check on the d table."""
    for D, N in ((2, 3), (3, 3), (4, 3)):
        lat = b.Lattice(D, N)
        for p in range(D):
            a, c = lat.random(p), lat.random(p + 1)
            lhs = float((np.asarray(b.d(a)) * np.asarray(c)).sum())
            rhs = float((np.asarray(a) * np.asarray(b.delta(c))).sum())
            assert not np.isclose(lhs, 0), f'D={D} p={p}: adjointness is vacuous'
            assert np.isclose(lhs, rhs), f'D={D} p={p}: {lhs} vs {rhs}'


def test_delta_squared_is_zero():
    lat = b.Lattice(4, 3)
    for p in range(2, 5):
        a = lat.random(p)
        da = b.delta(a)
        assert not np.allclose(np.asarray(da), 0)
        assert np.allclose(np.asarray(b.delta(da)), 0)


def test_cup_adds_degrees():
    lat = b.Lattice(4, 3)
    assert b.cup(lat.random(1), lat.random(2)).degree == 3


def test_cup_is_bilinear():
    lat = b.Lattice(3, 3)
    for p, q in ((0, 1), (1, 1), (1, 2), (2, 1)):
        if p + q > 3:
            continue
        a1, a2, c = lat.random(p), lat.random(p), lat.random(q)
        lhs = b.cup(a1 + a2, c)
        rhs = np.asarray(b.cup(a1, c)) + np.asarray(b.cup(a2, c))
        assert not np.allclose(rhs, 0), f'{p} cup {q} vanished identically'
        assert np.allclose(np.asarray(lhs), rhs)
        c1, c2 = lat.random(q), lat.random(q)
        lhs = b.cup(a1, c1 + c2)
        rhs = np.asarray(b.cup(a1, c1)) + np.asarray(b.cup(a1, c2))
        assert not np.allclose(rhs, 0), f'{p} cup {q} vanished identically'
        assert np.allclose(np.asarray(lhs), rhs)


def test_cup_is_associative():
    lat = b.Lattice(4, 3)
    for p, q, r in ((1, 1, 1), (1, 1, 2), (2, 1, 1), (0, 2, 2)):
        if p + q + r > 4:
            continue
        a, c, e = lat.random(p), lat.random(q), lat.random(r)
        lhs = b.cup(b.cup(a, c), e)
        rhs = b.cup(a, b.cup(c, e))
        assert not np.allclose(np.asarray(lhs), 0)
        assert np.allclose(np.asarray(lhs), np.asarray(rhs)), f'{p},{q},{r}'


def test_leibniz():
    """d(a cup c) = da cup c + (-1)^p a cup dc."""
    lat = b.Lattice(4, 3)
    for p, q in ((0, 1), (1, 1), (1, 2), (2, 1)):
        if p + q + 1 > 4:
            continue
        a, c = lat.random(p), lat.random(q)
        lhs = b.d(b.cup(a, c))
        rhs = b.cup(b.d(a), c) + (-1)**p * b.cup(a, b.d(c))
        assert not np.allclose(np.asarray(lhs), 0)
        assert np.allclose(np.asarray(lhs), np.asarray(rhs)), f'{p},{q}'


def test_cup_with_a_zero_cochain_is_multiplication():
    lat = b.Lattice(3, 3)
    phi, a = lat.random(0), lat.random(1)
    lhs = b.cup(phi, a)
    rhs = np.asarray(a) * np.asarray(phi)[0] * lat.support(1)
    assert not np.allclose(rhs, 0), '0 cup 1 vanished identically'
    assert np.allclose(np.asarray(lhs), rhs)


def test_wedge_is_cup():
    assert b.wedge is b.cup


def _plaquette_families(lat, mu, nu, z):
    """The blue and orange triangle slots at the centre of plaquette (mu,nu) at z.

    Returns (blue, orange), each a list of (type_index, anchor_tuple, sign2d),
    where sign2d is the orientation of the triangle inside its own square.

    Two fixes on top of the naive per-plane construction, both needed to make
    P . P* = +1 for every plaquette (note section on the intersection form):

    - The **anchor** of a blue triangle is its own v0, reached from the
      centre xi_P (= its own v2) by subtracting that type's offset. The
      orange triangle is different: xi_P is its own v0 (the anchor and the
      centre coincide -- xi_P already has even coordinates in the
      complementary directions), so no offset subtraction applies there.
      Subtracting it anyway (as a first attempt did) reads the orange flux
      off a location outside its support -- always zero -- which is exactly
      the kind of vacuous 0 == 0 pass the non-degeneracy guards below exist
      to catch.
    - The dual (orange) family additionally needs the global orientation
      factor epsilon_{mu,nu,o1,o2} (o1 < o2 the complementary directions):
      the sign of the permutation (mu, nu, o1, o2) of (0, 1, 2, 3). Without
      it, planes related by an odd permutation (e.g. (mu,nu) = (1,3)) come
      out with the wrong overall sign relative to (0,1) or (2,3).
    """
    s = b._structures(4)
    N = lat.N
    others = [k for k in range(4) if k not in (mu, nu)]
    centre = tuple((2 * z[k] + (1 if k in (mu, nu) else 0)) % (2 * N)
                   for k in range(4))
    perm = (mu, nu) + tuple(others)
    epsilon = -1 if sum(1 for i in range(4) for j in range(i + 1, 4)
                         if perm[i] > perm[j]) % 2 else 1

    def family(pair, anchor_is_v2, extra_sign):
        out = []
        for first, second in ((pair[0], pair[1]), (pair[1], pair[0])):
            for sa, sb in product((+1, -1), repeat=2):
                steps = (((first, sa),), ((second, sb),))
                t = s.type_index[2][steps]
                if anchor_is_v2:
                    offset = s.offsets[2][t][2]
                    anchor = tuple((centre[k] - offset[k]) % (2 * N) for k in range(4))
                else:
                    anchor = centre
                # orientation of the triangle inside its own plane: the sign of
                # the 2x2 determinant of its two edge vectors, restricted to the
                # plane's two directions in increasing order.
                plane = sorted(pair)
                W = s.offsets[2][t][1:][:, plane]       # (2,2)
                sign2d = int(round(np.sign(np.linalg.det(W)))) * extra_sign
                out.append((t, anchor, sign2d))
        return out

    return family((mu, nu), True, 1), family(tuple(others), False, epsilon)


def test_sixty_four_simplices_meet_at_a_plaquette_centre():
    """Note section on the intersection form: the fine four-simplices whose
    rank-two vertex v2 is a given plaquette centre are *exactly* the 8 x 8
    blue-triangle x orange-triangle pairs.

    Brute-forced against the module rather than counted off the helper: a
    length check on ``_plaquette_families``' own output is true by that
    helper's construction and references nothing in ``barycentric``.  Here
    every one of the 384 degree-4 types is tested for whether its v2 lands on
    the centre, and the resulting slot set is compared with the pair set."""
    lat = b.Lattice(4, 3)
    s = b._structures(4)
    M = 2 * lat.N
    mu, nu, z = 0, 1, (1, 1, 1, 1)
    centre = tuple((2 * z[k] + (1 if k in (mu, nu) else 0)) % M for k in range(4))

    # every (type, anchor) slot in support(4) whose v2 is the centre
    found = set()
    for t4 in range(s.type_count[4]):
        offset = s.offsets[4][t4][2]
        anchor = tuple((centre[k] - offset[k]) % M for k in range(4))
        if lat.support(4)[(t4,) + anchor]:
            found.add((t4, anchor))

    blue, orange = _plaquette_families(lat, mu, nu, z)
    assert len(blue) == 8 and len(orange) == 8
    pairs = {(s.type_index[4][s.step_types[2][tb] + s.step_types[2][to]], xb)
             for tb, xb, _ in blue for to, _, _ in orange}
    assert len(pairs) == 64
    assert found == pairs, (
        f'{len(found)} four-simplices have v2 at the centre, against '
        f'{len(pairs)} blue x orange pairs; symmetric difference '
        f'{len(found ^ pairs)}')


def test_charge_factorizes_into_primal_and_dual_flux():
    """Note eq. (13): sum of the 64 charges at b(P) is Phi(P) Phi(P*)."""
    rng = np.random.default_rng(2026)
    lat = b.Lattice(4, 3)
    s = b._structures(4)
    n = lat.random_integer(1)
    f = b.d(n)
    q = lat.charge(f)
    raw_f, raw_q = np.asarray(f), np.asarray(q)

    sampled_products = []
    for (mu, nu) in ((0, 1), (1, 3), (2, 3)):
        for _ in range(3):
            z = tuple(rng.integers(0, lat.N, 4))
            blue, orange = _plaquette_families(lat, mu, nu, z)
            Phi_P  = sum(sg * raw_f[(t,) + xi] for t, xi, sg in blue)
            Phi_Ps = sum(sg * raw_f[(t,) + xi] for t, xi, sg in orange)
            total = 0
            for tb, xb, _ in blue:
                for to, _, _ in orange:
                    # the 4-simplex whose first two steps are the blue triangle's
                    # and whose last two are the orange triangle's
                    steps = s.step_types[2][tb] + s.step_types[2][to]
                    t4 = s.type_index[4][steps]
                    total += raw_q[(t4,) + xb]
            assert total == Phi_P * Phi_Ps, f'({mu},{nu}) at {z}'
            sampled_products.append((Phi_P, Phi_Ps))

    assert any(Phi_P != 0 and Phi_Ps != 0 for Phi_P, Phi_Ps in sampled_products), (
        'every sampled (Phi_P, Phi_Ps) pair had a zero factor -- the sample '
        'never exercised the non-trivial factorization, so this test cannot '
        'tell eq. (13) from a vacuous 0 == 0.')


def test_constraint_collapses_to_the_occupancy_rule():
    """Note eq. (14): q = 0 on all 64 iff the blue family or the orange family
    carries no flux at all.

    Random dn flux is dense: with N = 3 in D = 4 it essentially never leaves
    either family exactly fluxless, so the 20 random trials below only ever
    land on the all_zero == False branch of the iff (checked, not assumed --
    see the guard after the loop). The other branch -- one family exactly
    fluxless, forcing all 64 charges to vanish -- is exercised separately, by
    construction, on a general (not necessarily closed) 2-cochain: this is
    the algebraic content of eq. (14), that an outer product Phi(P) Phi(P*)
    vanishes entrywise iff one of its two factors vanishes, and it holds for
    any 2-cochain since `Lattice.charge` never requires closedness. It is
    not a claim that some dn configuration exhibits this branch.
    """
    rng = np.random.default_rng(7)
    lat = b.Lattice(4, 3)
    s = b._structures(4)
    outcomes = set()
    for trial in range(20):
        n = lat.random_integer(1, low=-1, high=2)
        f = b.d(n)
        q = lat.charge(f)
        raw_f, raw_q = np.asarray(f), np.asarray(q)
        z = tuple(rng.integers(0, lat.N, 4))
        blue, orange = _plaquette_families(lat, 0, 2, z)
        blue_flux   = [raw_f[(t,) + xi] for t, xi, _ in blue]
        orange_flux = [raw_f[(t,) + xi] for t, xi, _ in orange]
        charges = []
        for tb, xb, _ in blue:
            for to, _, _ in orange:
                t4 = s.type_index[4][s.step_types[2][tb] + s.step_types[2][to]]
                charges.append(raw_q[(t4,) + xb])
        all_zero = all(c == 0 for c in charges)
        occupancy = (all(x == 0 for x in blue_flux)
                     or all(x == 0 for x in orange_flux))
        assert all_zero == occupancy, f'trial {trial}'
        outcomes.add(all_zero)

    assert False in outcomes, (
        f'never observed all_zero == False across 20 random dn trials -- '
        f'outcomes = {outcomes} -- so this loop is not exercising the iff '
        'at all.')

    # The complementary branch: zero out one family's eight slots by hand on
    # a general 2-cochain and check that all 64 charges at that centre
    # vanish, with the *other* family checked nonzero so the case is not
    # merely f = 0.
    for zero_family in ('blue', 'orange'):
        f = lat.random_integer(2, low=-2, high=3)
        raw_f = np.asarray(f)
        z = tuple(rng.integers(0, lat.N, 4))
        blue, orange = _plaquette_families(lat, 0, 2, z)
        zeroed, kept = (blue, orange) if zero_family == 'blue' else (orange, blue)
        for t, xi, _ in zeroed:
            raw_f[(t,) + xi] = 0
        kept_flux = [raw_f[(t,) + xi] for t, xi, _ in kept]
        assert any(x != 0 for x in kept_flux), (
            f'zeroing the {zero_family} family left the kept family also '
            'all zero -- this constructed case only tests f = 0, not the '
            'occupancy rule.')
        raw_q = np.asarray(lat.charge(f))
        charges = []
        for tb, xb, _ in blue:
            for to, _, _ in orange:
                t4 = s.type_index[4][s.step_types[2][tb] + s.step_types[2][to]]
                charges.append(raw_q[(t4,) + xb])
        assert all(c == 0 for c in charges), (
            f'zeroing the {zero_family} family did not zero all 64 charges '
            f'at the plaquette centre')


def test_blind_links_never_enter_the_charge():
    """Note eq. (17): classes (0,3),(0,4),(1,3),(1,4) jump the plaquette rank."""
    lat = b.Lattice(4, 3)
    blind = {(0, 3), (0, 4), (1, 3), (1, 4)}
    blind_idx = [i for i, c in enumerate(lat.link_classes) if c in blind]
    mask = np.isin(lat.link_class, blind_idx)
    n = lat.random_integer(1)
    bl = lat.zeros(1, dtype=int)
    np.asarray(bl)[mask] = np.random.randint(-2, 3, int(mask.sum()))
    q0 = np.asarray(lat.charge(b.d(n)))
    q1 = np.asarray(lat.charge(b.d(n + bl)))
    assert not np.allclose(np.asarray(bl), 0)
    assert not (q0 == 0).all(), (
        'the reference charge q0 vanished identically -- q0 == q1 would then '
        'hold even if charge() simply returned zeros.')
    assert (q0 == q1).all()


def test_valid_accepts_the_vacuum_and_rejects_a_charged_configuration():
    """The note's own deterministic witness: put n = 1 on the edge 01 and on
    the edge 34 of one ordered four-simplex sigma.  Then
    f_012 = n_01 = 1 and f_234 = n_34 = 1, so q = eps_sigma = +-1 there, and
    Sum |q| = 2 over the whole lattice.

    This is a check against the collaborator's text rather than against
    ``valid``'s own definition -- the earlier version guarded on
    ``charge(d(n)) != 0`` and then asserted ``not valid(n)``, which is what
    ``valid`` *means*, and which moreover skipped itself whenever the random
    draw happened to be uncharged."""
    lat = b.Lattice(4, 3)
    s = b._structures(4)
    M = 2 * lat.N
    assert lat.valid(lat.zeros(1, dtype=int))

    # an ordered four-simplex: the ascending staircase 0 -> 1 -> 2 -> 3 at the
    # origin, whose five vertices are xi, xi+e0, xi+e0+e1, ... , xi+e0+e1+e2+e3
    steps = tuple(((mu, +1),) for mu in range(4))
    sigma = s.type_index[4][steps]
    anchor = (0, 0, 0, 0)
    assert lat.support(4)[(sigma,) + anchor] == 1

    # edge 01 is the first step out of v0 = the anchor; edge 34 is the fourth
    # step out of v3
    t01 = s.type_index[1][(steps[0],)]
    t34 = s.type_index[1][(steps[3],)]
    v3 = tuple((anchor[k] + s.offsets[4][sigma][3][k]) % M for k in range(4))
    assert lat.support(1)[(t01,) + anchor] == 1
    assert lat.support(1)[(t34,) + v3] == 1

    n = lat.zeros(1, dtype=int)
    np.asarray(n)[(t01,) + anchor] = 1
    np.asarray(n)[(t34,) + v3] = 1

    q = np.asarray(lat.charge(b.d(n)))
    epsilon = int(np.asarray(lat.orientation_sign).reshape(-1)[sigma])
    assert q[(sigma,) + anchor] == epsilon
    assert int(np.abs(q).sum()) == 2
    assert set(np.unique(q[q != 0])) <= {-1, 1}
    assert not lat.valid(n)


# ---------------------------------------------------------------------------
# The point group
# ---------------------------------------------------------------------------

def test_reflect_and_permute_preserve_support_and_degree():
    lat = b.Lattice(3, 3)
    for p in range(4):
        a = lat.random(p)
        for flips in ((0,), (0, 2)):
            r = b.reflect(a, flips)
            assert r.degree == p
            assert (np.asarray(r)[lat.support(p) == 0] == 0).all()
        pm = b.permute(a, (1, 2, 0))
        assert pm.degree == p
        assert (np.asarray(pm)[lat.support(p) == 0] == 0).all()


def test_reflect_is_an_involution():
    lat = b.Lattice(3, 3)
    a = lat.random(2)
    once = np.asarray(b.reflect(b.reflect(a, (1,)), (1,)))
    assert not np.allclose(np.asarray(a), 0), (
        'random(2) came out identically zero -- the comparison below would '
        'pass vacuously.')
    assert np.allclose(once, np.asarray(a))


def test_permute_composes():
    lat = b.Lattice(3, 3)
    a = lat.random(1)
    once  = b.permute(b.permute(a, (1, 2, 0)), (1, 2, 0))
    twice = b.permute(a, (2, 0, 1))
    assert not np.allclose(np.asarray(once), 0), (
        'random(1) came out identically zero -- the comparison below would '
        'pass vacuously.')
    assert np.allclose(np.asarray(once), np.asarray(twice))


def test_point_group_order_is_measured_not_assumed():
    """Note section 3.1: the FULL hypercubic group acts, order 2^D D! -- 384 in
    D = 4, against the cubical model's 48.  Count distinct actions, so a
    transformation that silently collapses onto another is caught.

    .. note ::
        D = 4 is included in the sweep itself -- not asserted as bare
        arithmetic beside it -- so the headline 384 is measured, not assumed."""
    for D, N in ((2, 3), (3, 2), (4, 3)):
        lat = b.Lattice(D, N)
        a = lat.random(1)
        seen = set()
        for perm in permutations(range(D)):
            for bits in product((0, 1), repeat=D):
                flips = tuple(k for k in range(D) if bits[k])
                moved = b.permute(b.reflect(a, flips), perm)
                seen.add(np.asarray(moved).tobytes())
        assert len(seen) == 2**D * factorial(D), f'D={D}: {len(seen)}'


def test_whole_point_group_preserves_the_link_class_multiset():
    """The histogram is taken of the *nonzero support* of the transformed
    data, not of the static lattice structure -- so this actually depends on
    what permute/reflect do to n, rather than holding unconditionally."""
    lat = b.Lattice(4, 3)
    n = lat.random_integer(1)
    n_classes = lat.link_class[(lat.link_class >= 0) & (np.asarray(n) != 0)]
    baseline = np.bincount(n_classes.ravel(), minlength=len(lat.link_classes))
    assert baseline.sum() > 0, (
        'n has no nonzero entries on real link slots -- the multiset check '
        'below would hold vacuously for the zero configuration.')
    order = 0
    for perm in permutations(range(4)):
        for bits in product((0, 1), repeat=4):
            flips = tuple(k for k in range(4) if bits[k])
            moved = b.permute(b.reflect(n, flips), perm)
            raw_moved = np.asarray(moved)
            assert (raw_moved[lat.support(1) == 0] == 0).all()
            moved_classes = lat.link_class[(lat.link_class >= 0) & (raw_moved != 0)]
            histogram = np.bincount(moved_classes.ravel(), minlength=len(lat.link_classes))
            assert (histogram == baseline).all()
            order += 1
    assert order == 384


def _signature(perm):
    """The sign of a permutation, by inversion count."""
    return (-1)**sum(1 for i in range(len(perm))
                     for j in range(i + 1, len(perm)) if perm[i] > perm[j])


def test_whole_point_group_acts_covariantly_on_the_charge():
    """The headline, sharpened: all 384 elements, not the cubical model's 48,
    and covariance rather than mere membership.

    The charge is a top-degree density, so a signed coordinate permutation
    $g$ carries it with the determinant:

        q(g . n)  ==  det(g) * g . q(n).

    That is strictly stronger than "$g$ preserves the constraint set", which
    it implies (if $q(n) = 0$ everywhere then so is $q(g.n)$), and it is the
    actual statement of the note's claim (b).  The membership version this
    replaced tested a single-link configuration whose charge vanishes for a
    structural reason -- the cup-square's two triangles 012 and 234 meet only
    at the vertex v2, so no *single* edge can carry flux into both, hence
    q == 0 identically -- which made it a near-vacuum: it still passed with
    ``reflect`` replaced by the identity."""
    lat = b.Lattice(4, 3)
    n = lat.random_integer(1)
    q = lat.charge(b.d(n))
    assert int((np.asarray(q) != 0).sum()) > 0, (
        'the reference charge vanished identically -- covariance would then '
        'be 0 == 0 for every group element.')

    order = 0
    for perm in permutations(range(4)):
        for bits in product((0, 1), repeat=4):
            flips = tuple(k for k in range(4) if bits[k])
            det = _signature(perm) * (-1)**len(flips)
            moved = b.permute(b.reflect(n, flips), perm)
            lhs = np.asarray(lat.charge(b.d(moved)))
            rhs = det * np.asarray(b.permute(b.reflect(q, flips), perm))
            assert (lhs == rhs).all(), f'perm={perm} flips={flips}'
            order += 1
    assert order == 384


# ---------------------------------------------------------------------------
# The half-diagonal translation
# ---------------------------------------------------------------------------

def test_only_the_all_ones_parity_shift_preserves_the_complex():
    """Verified by enumeration of both the fine link set AND the fine
    triangle set, because several mixed shifts preserve individual links
    (or triangles) while failing on the link/triangle set as a whole.

    Enumerated at D = 2 (N = 3), D = 3 (N = 2) and D = 4 (N = 2) -- the
    physical dimension included, so the claim is checked where the model
    lives and not only extrapolated from below.  D = 4 at N = 2 is 3840 fine
    links and 12800 fine triangles and costs well under a second."""
    def fine_links(D, N):
        M = 2 * N
        links = set()
        for xi in product(range(M), repeat=D):
            even = [mu for mu in range(D) if xi[mu] % 2 == 0]
            for k in range(1, len(even) + 1):
                for chosen in product(even, repeat=k):
                    if len(set(chosen)) != k:
                        continue
                    for signs in product((+1, -1), repeat=k):
                        eta = list(xi)
                        for mu, s in zip(chosen, signs):
                            eta[mu] = (eta[mu] + s) % M
                        links.add(frozenset((xi, tuple(eta))))
        return links

    def fine_triangles(D, N):
        """A triangle is a two-step staircase: a first signed step out of
        the anchor's even directions to v1, then a second signed step out
        of whichever directions are still even at v1 (i.e. not touched by
        the first step) to v2.  Represented as the unordered vertex triple
        {xi, v1, v2}, exactly as fine_links represents a link by the
        unordered pair {xi, eta} -- so no separate reversal handling is
        needed, a frozenset absorbs any reordering."""
        M = 2 * N
        triangles = set()
        for xi in product(range(M), repeat=D):
            even0 = [mu for mu in range(D) if xi[mu] % 2 == 0]
            for k1 in range(1, len(even0) + 1):
                for chosen1 in product(even0, repeat=k1):
                    if len(set(chosen1)) != k1:
                        continue
                    for signs1 in product((+1, -1), repeat=k1):
                        v1 = list(xi)
                        for mu, s in zip(chosen1, signs1):
                            v1[mu] = (v1[mu] + s) % M
                        remaining = [mu for mu in even0 if mu not in chosen1]
                        for k2 in range(1, len(remaining) + 1):
                            for chosen2 in product(remaining, repeat=k2):
                                if len(set(chosen2)) != k2:
                                    continue
                                for signs2 in product((+1, -1), repeat=k2):
                                    v2 = list(v1)
                                    for mu, s in zip(chosen2, signs2):
                                        v2[mu] = (v2[mu] + s) % M
                                    triangles.add(frozenset(
                                        (xi, tuple(v1), tuple(v2))))
        return triangles

    def shifted_by(cells, v, M, D):
        return {frozenset(tuple((x[k] + v[k]) % M for k in range(D))
                          for x in cell) for cell in cells}

    for D, N in ((2, 3), (3, 2), (4, 2)):
        L = fine_links(D, N)
        T = fine_triangles(D, N)
        M = 2 * N
        expected = [(0,) * D, (1,) * D]

        survivors_links = [v for v in product((0, 1), repeat=D)
                            if shifted_by(L, v, M, D) == L]
        assert survivors_links == expected, (
            f'D={D}: link-set survivors {survivors_links}')

        survivors_triangles = [v for v in product((0, 1), repeat=D)
                                if shifted_by(T, v, M, D) == T]
        assert survivors_triangles == expected, (
            f'D={D}: triangle-set survivors {survivors_triangles}')

        survivors_both = [v for v in product((0, 1), repeat=D)
                          if shifted_by(L, v, M, D) == L
                          and shifted_by(T, v, M, D) == T]
        assert survivors_both == expected, f'D={D}: {survivors_both}'


def test_tau_reverses_rank():
    lat = b.Lattice(4, 3)
    shifted = np.roll(lat.site_rank, (1, 1, 1, 1), axis=(0, 1, 2, 3))
    assert (shifted == 4 - lat.site_rank).all()


def test_tau_maps_link_class_antidiagonally():
    """Note Figure 5: (r,s) -> (D-s, D-r)."""
    lat = b.Lattice(4, 3)
    idx = {c: i for i, c in enumerate(lat.link_classes)}
    for source, target in (((0, 1), (3, 4)), ((0, 3), (1, 4)),
                           ((0, 4), (0, 4)), ((1, 3), (1, 3))):
        marked = lat.zeros(1, dtype=float)
        np.asarray(marked)[lat.link_class == idx[source]] = 1.0
        moved = b.tau(marked)
        carried = np.asarray(moved) != 0
        assert carried.any(), f'{source}: tau produced no nonzero output'
        assert (lat.link_class[carried] == idx[target]).all(), f'{source}->{target}'


def test_tau_squared_is_the_unit_diagonal_translation():
    """Composed, not short-circuited.

    ``tau(a, 2)`` does not compute anything: it returns
    ``translate(a, (1,)*D)`` from an explicit even-power branch, so comparing
    *that* against the same call is X == X and cannot fail whatever the odd
    branch does.  Applying the odd branch twice is the check with teeth --
    it is the only assertion here that constrains tau's roll direction, and
    with the roll negated it fails."""
    lat = b.Lattice(4, 3)
    a = lat.random(2)
    once = b.tau(a, 1)
    assert not np.allclose(np.asarray(once), 0), (
        'tau(a) vanished identically -- the comparisons below would pass '
        'vacuously.')
    twice = np.asarray(b.tau(once, 1))
    assert np.allclose(twice, np.asarray(b.translate(a, (1, 1, 1, 1))))
    # and the even-power shortcut agrees with the honest composition
    assert np.allclose(np.asarray(b.tau(a, 2)), twice)


def test_tau_moves_a_marked_simplex_to_the_predicted_slot_with_the_predicted_sign():
    r"""The direct statement of what tau does to one simplex.

    Mark the single ``(type, anchor)`` slot of one fine p-simplex sigma and
    apply tau.  Its image is the same vertex set translated by the
    half-diagonal --- which in doubled coordinates is +1 in *every* direction
    --- with the vertex order reversed, because rank reverses.  So

    - the image's anchor (its new lowest-rank vertex) is sigma's *top*
      vertex translated: ``anchor + offsets[p][t][p] + 1``;
    - the image's type is the one whose offset table is sigma's read
      backwards from that new anchor, ``offsets[p][t][::-1] - offsets[p][t][p]``;
    - the value carries ``(-1)**(p(p+1)/2)``, the sign of reversing p+1
      vertices: -1 at p = 1 and p = 2, +1 at p = 3.

    The prediction is built from ``offsets`` (independently tested in
    ``test_offsets_are_cumulative_signed_unit_steps``) and from the closed
    form for the sign, *not* from ``_Structures.half_diagonal``, so this
    fails if either the roll direction or the sign is changed."""
    lat = b.Lattice(4, 3)
    s = b._structures(4)
    M = 2 * lat.N
    for p, expected_sign in ((1, -1), (2, -1), (3, +1)):
        assert expected_sign == (-1)**(p * (p + 1) // 2)
        offsets = s.offsets[p]
        # a few types per degree, each at an anchor where it is a real simplex
        for t in (0, s.type_count[p] // 3, s.type_count[p] - 1):
            anchors = np.argwhere(np.asarray(lat.support(p)[t]) == 1)
            assert len(anchors), f'type {t} of degree {p} is never supported'
            anchor = tuple(int(x) for x in anchors[len(anchors) // 2])

            marked = lat.zeros(p, dtype=int)
            np.asarray(marked)[(t,) + anchor] = 1
            moved = np.asarray(b.tau(marked))

            top = offsets[t][p]
            predicted_anchor = tuple((anchor[k] + top[k] + 1) % M
                                     for k in range(4))
            target = offsets[t][::-1] - top
            candidates = [u for u in range(s.type_count[p])
                          if (offsets[u] == target).all()]
            assert len(candidates) == 1, (
                f'degree {p} type {t}: {len(candidates)} types match the '
                'reversed staircase')
            predicted_type = candidates[0]

            slot = (predicted_type,) + predicted_anchor
            assert lat.support(p)[slot] == 1, (
                f'degree {p} type {t}: the predicted image slot is not a '
                'real simplex, so the assertions below could not be met by '
                'any placement')
            assert moved[slot] == expected_sign, (
                f'degree {p} type {t}: tau put {moved[slot]} at the '
                f'predicted slot, expected {expected_sign}')
            assert int((moved != 0).sum()) == 1, (
                f'degree {p} type {t}: tau of a single marked simplex has '
                f'{int((moved != 0).sum())} nonzero entries, expected 1')


def test_tau_acts_covariantly_on_the_charge():
    """tau is an orientation-preserving translation of the complex, so the
    charge -- a top-degree density -- goes along with it unsigned:
    ``q(tau n) == tau q(n)``.  (The rank reversal costs
    ``(-1)**(4*5/2) = +1`` at p = 4.)"""
    lat = b.Lattice(4, 3)
    n = lat.random_integer(1)
    q = lat.charge(b.d(n))
    assert int((np.asarray(q) != 0).sum()) > 0, (
        'the reference charge vanished identically -- the comparison below '
        'would be 0 == 0.')
    lhs = np.asarray(lat.charge(b.d(b.tau(n))))
    rhs = np.asarray(b.tau(q))
    assert (lhs == rhs).all()


def test_tau_preserves_the_constraint_set():
    lat = b.Lattice(4, 3)
    n = lat.zeros(1, dtype=int)
    np.asarray(n)[0][(2, 2, 2, 2)] = 1
    n = n * lat.support(1)
    assert not np.allclose(np.asarray(n), 0), (
        'the single-link excitation vanished after masking by support(1) -- '
        'the test below would only be checking that tau preserves the '
        'vacuum.')
    assert lat.valid(n)
    assert lat.valid(b.tau(n))


def test_translate_diagonal_reduces_to_tau_and_a_translation():
    lat = b.Lattice(4, 3)
    a = lat.random(2)
    for signs in ((1, 1, 1, 1), (1, -1, 1, -1), (-1, -1, -1, -1)):
        integer = tuple((signs[k] - 1) // 2 for k in range(4))
        lhs = np.asarray(b.translate_diagonal(a, signs))
        assert not np.allclose(lhs, 0), (
            f'translate_diagonal(a, {signs}) vanished identically -- the '
            'comparison below would pass vacuously.')
        assert np.allclose(lhs, np.asarray(b.translate(b.tau(a), integer)))


def test_translate_diagonal_with_even_steps_is_a_plain_translation():
    lat = b.Lattice(4, 3)
    a = lat.random(1)
    lhs = np.asarray(b.translate_diagonal(a, (1, 1, 1, 1), steps=2))
    assert not np.allclose(lhs, 0), (
        'translate_diagonal(a, ..., steps=2) vanished identically -- the '
        'comparison below would pass vacuously.')
    assert np.allclose(lhs, np.asarray(b.translate(a, (1, 1, 1, 1))))


def test_two_dimensional_figure_one():
    """Note Figure 1: a square is cut into eight fine triangles, three ranks."""
    lat = b.Lattice(2, 3)
    assert lat.count(2) / 3**2 == 8
    assert sorted(set(lat.site_rank.ravel())) == [0, 1, 2]
    assert np.isclose((lat.measure(2) * lat.support(2)).sum(), 9)
