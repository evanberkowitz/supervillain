import pytest
import numpy as np
from supervillain.analysis.autocorrelation import autocorrelation_time

# The idea behind these tests is that if we do uncorrelated random draws
# then the autocorrelation time should be absolutely minimal, which is τ=1.
# Prior to this test (and its associated fix), the minimal autocorrelation time
# was 2, which is conventionally wrong and which caused me a lot of confusion.


def test_autocorrelation_time():
    rng = np.random.default_rng(12345)
    data = rng.normal(size=100000)
    assert autocorrelation_time(data) == 1

def test_autocorrelation_time_with_mean():
    rng = np.random.default_rng(12345)
    data = rng.normal(size=100000, loc=1)
    assert autocorrelation_time(data, mean=1) == 1
    
    