#!/usr/bin/env python

import pytest
import harness

@harness.skip_on(NotImplementedError, 'Not implemented')
def test_skip():
    raise NotImplementedError()

@pytest.mark.xfail
@harness.skip_on(NotImplementedError, 'Not implemented')
def test_fail():
    raise ValueError()
