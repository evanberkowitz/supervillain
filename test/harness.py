import pytest
from functools import wraps
import supervillain

def skip_on(exception, explanation):

    def the_decorator(f):

        @wraps(f)
        def decorated_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exception:
                pytest.skip(explanation)

        return decorated_f

    return the_decorator

def for_each_test_ensemble(f):

    @pytest.mark.parametrize('W', (1,2,))
    @pytest.mark.parametrize('kappa', (0.4, 0.5, 0.6))
    @pytest.mark.parametrize('N', (3, 4, 7, 8))
    @wraps(f)
    def decorated_f(*args, **kwargs):
        return f(*args, **kwargs)

    return decorated_f

def for_each_observable(f):

    @pytest.mark.parametrize('observable', supervillain.observables.keys())
    @wraps(f)
    def decorated_f(*args, **kwargs):
        return f(*args, **kwargs)

    return decorated_f

def measure_without_inline(ensemble, observable):
    # If the observable is measured inline we need to prevent the short-circuiting.
    if observable in ensemble.configuration.fields:
        # Temporarily store inline observables...
        tmp = ensemble.configuration.fields[observable]
        del ensemble.configuration.fields[observable]
        # ... measure the desired observable
        value = getattr(ensemble, observable)
        # ... and restore the inline measurement.
        ensemble.configuration.fields[observable] = tmp
    # But if the observable wasn't measured inline, measure it!
    else:
        value = getattr(ensemble, observable)

    return value


