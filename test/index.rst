
Tests
=====

We use `pytest`_ to implement tests.  From the repository's root directory you can do

.. code-block::

    > pytest .
    ==================================== test session starts ====================================
    platform darwin -- Python 3.11.8, pytest-8.1.1, pluggy-1.4.0
    rootdir: /Users/evanberkowitz/physics/supervillain
    configfile: pyproject.toml
    plugins: anyio-4.3.0, cov-4.1.0
    collected 962 items

    test/test_compare-reference-observables.py .......................................... [  4%]
    ......                                                                                [  4%]
    test/test_ensemble-extension.py ................................................      [  9%]
    test/test_equivalence-class-v.py .................................................... [ 15%]
    ..................................................................................... [ 24%]
    ..................................................................................... [ 33%]
    ..................................................................................... [ 41%]
    ..................................................................................... [ 50%]
    ........................................                                              [ 54%]
    test/test_gauge-invariance.py ....................................................... [ 60%]
    ..................................................................................... [ 69%]
    ..................................................................................... [ 78%]
    ..................................................................................... [ 87%]
    ..ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss........... [ 95%]
    .....................................                                                 [ 99%]
    test/test_harness.py sx                                                               [100%]

    ================== 888 passed, 73 skipped, 1 xfailed in 290.62s (0:04:50) ===================

Some tests are skipped on purpose (for example, if an action does not have a particular observable implemented) or `xfail`_\ed purposefully; only actual failures should be of concern.  You can get a more detailed report with ``-v``.

At first the tests may seem very slow.
For meaningful tests different ensembles are generated.
However, these ensembles are cached, so that the tests are slow to start but go faster as later Monte Carlo is not needed.

.. _pytest : https://docs.pytest.org/en/8.1.x/
.. _xfail : https://docs.pytest.org/en/8.1.x/reference/reference.html#pytest.mark.xfail

