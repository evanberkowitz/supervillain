def test_sweep_runs_small():
    from benchmark.form_kernels import sweep
    rows = sweep(Ds=(2, 3), Ns=(3,))
    assert rows, "sweep produced no rows"
    for r in rows:
        assert {"D", "N", "op", "ref_us", "kernel_us", "speedup"} <= set(r)
        assert r["kernel_us"] > 0
