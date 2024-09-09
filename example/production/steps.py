#!/usr/bin/env python

import h5py as h5

import supervillain
from supervillain.performance import Timer
from supervillain.h5 import Data

def progress(iterable, **kwargs):
    r'''
    Like `tqdm <https://tqdm.github.io/docs/tqdm/#tqdm-objects>`_, but requires the iterable.

    The default progress bar is a no-op that forwards the iterable.  You can overwrite it simply.

    .. code:: python

        from tqdm import tqdm
        import steps
        steps.progress=tqdm
    '''
    return iterable

import logging
logger = logging.getLogger(__name__)

# The job of the input file is to construct a pandas dataframe with rows that describe
# the different ensembles we wish to include in our computation.
# The computation is made out of different steps.

####
#### Infrastructure
####

# It would probably be good to learn to use a proper make-like directed acyclic graph
# that guarantees that steps only need to be run once.
# But our computational steps will either be extremely cheap, or so damned expensive that we cached them on disk.

class Step:
    # A step has prerequisites given in the ingredients list.
    ingredients = dict()

    # To complete the step the ingredients must be prepared and ready.
    @classmethod
    def prep(cls, row):
        with Timer(logger.info, f'Preparing ingredients for {cls.__name__}'):
            return {key: i.of(row) for key, i in cls.ingredients.items()}

    # Each step provides its own step.of(row) method.
    # Its job is to actually accomplish the computational step.
    @classmethod
    def of(cls, row):
        raise NotImplementedError()

# Some of the steps are so costly that we store the results in an hdf5 file for later re-use.
def h5_cached(decorated_cls):

    try:
        target = getattr(decorated_cls, 'target')
        assert callable(target)
    except:
        raise ValueError(f'{decorated_cls.__name__} needs a target classmethod to be h5_cached.') from None

    # In that case we construct a new class
    class Curried(decorated_cls):

        # but we intercept the computational method, short-circuiting if the result exists
        # in the step's target.
        @classmethod
        def of(cls, row):
            f, path = cls.target(row)
            logger.info(f'Checking {f}/{path}... ')
            try:
                with h5.File(f, 'r') as file:
                    return supervillain.h5.Data.read(file[path])
            except:
                with Timer(logger.info, f'Constructing {cls.__name__}'):
                    result = decorated_cls.of(row)
                    try:
                        with h5.File(f, 'a') as file:
                            supervillain.h5.Data.write(file, path, result)
                    except Exception as e:
                        raise e from None

            return result

        @classmethod
        def delete_h5(cls, row):

            f, path = cls.target(row)

            try:
                with h5.File(f, 'a') as file:
                    del file[path]
            except Exception as e:
                raise e from NOne

    Curried.__name__ = decorated_cls.__name__

    return Curried

# In post-processing steps like plotting or other analysis it can be useful to just use whatever data exists
# without triggering a lengthy computation to ensure all conceivable data is available.
#
# In that case we can call a Possible(step).of(row),
def Possible(maybe_cls):

    class Curried(maybe_cls):

        @classmethod
        def of(cls, row):
            f, path = cls.target(row)
            try:
                # which will give the true value if it is available
                with h5.File(f, 'r') as file:
                    return supervillain.h5.Data.read(file[path])
            except Exception as e:
                # and will return None otherwise.
                return None

    Curried.__name__ = maybe_cls.__name__
    return Curried


####
#### Our actual computation
####

# Roughly speaking, the thing we really want to produce is bootstrap-resampled decorrelated ensembles.
# The steps that don't require any intensive computation conceptually come first, to be used as
# ingredients in later steps.

# Some of the computational steps are extremely easy, just wrapping the obvious supervillain call.
class Lattice(Step):

    @classmethod
    def of(cls, row):
        return supervillain.lattice.Lattice2D(row['N'])

# Some of the steps need to make some row-dependent decisions.
class Action(Step):
    
    # Here we finally see how the ingredients work.
    # To build an action we need the Lattice.
    # Below, in the of() method we access the computed ingredient in the cooked dictionary according to the step's ingredient key.
    ingredients = {
            'lattice': Lattice,
            }

    @classmethod
    def of(cls, row):

        # We make sure the ingredients are already cooked
        cooked = cls.prep(row)

        # And then do the computational step itself.
        return (supervillain.action.Villain if row['action'] == 'Villain' else supervillain.action.Worldline)(cooked['lattice'], row['kappa'], row['W'])

class Generator(Step):

    ingredients = {
            'action': Action
            }

    @classmethod
    def of(cls, row):

        S = cls.prep(row)['action']
        L = S.Lattice # We can pull out previously-computed objects rather than put them into the ingredient list.

        if row['action'] == 'Villain':
            return supervillain.generator.villain.Hammer(S, L.plaquettes)
        elif row['action'] == 'Worldline':
            return supervillain.generator.worldline.Hammer(S, L.sites)

        raise ValueError(f"Unknown action {row['action']}")


# Now we are ready to thermalize.
# The result of thermalization is an ensemble which knows its autocorrelation time .tau
# It is expensive (and requires real numerical effort) so we write it to disk, to be reused.
@h5_cached
class Thermalization(Step):

    ingredients = {
            'action': Action,
            'generator': Generator
            }

    @classmethod
    def target(cls, row):
        return row['thermalization storage'], row['path']

    @classmethod
    def of(cls, row):

        cooked = cls.prep(row)
        S = cooked['action']
        G = cooked['generator']

        E = supervillain.Ensemble(S).generate(row['thermalize'], G, start=row['start'], progress=progress)

        E.measure()
        tau = E.autocorrelation_time()
        logger.info(f'Pre-thermalization  τ={tau}')

        E = E.cut(row['thermalization cut'] * tau)
        tau = E.autocorrelation_time()
        logger.info(f'Post-thermalization τ={tau}')

        E.tau = tau

        return E


# We decorrelate in the generator itself rather than using .every
# by wrapping the generator in the KeepEvery combining generator
# with strides given by the thermalized tau.
class DecorrelatedGenerator(Step):

    ingredients = {
            'thermalization': Thermalization,
            'generator':   Generator,
            }

    @classmethod
    def of(cls, row):

        cooked = cls.prep(row)
        tau = cooked['thermalization'].tau

        return supervillain.generator.combining.KeepEvery(tau, cooked['generator'])

# With the decorrelated generator in hand we can produce the actual ensemble we will analyze later.
@h5_cached
class Ensemble(Step):

    ingredients = {
            'action': Action,
            'generator': DecorrelatedGenerator,
            'thermalization': Thermalization,
            }

    @classmethod
    def target(cls, row):
        return row['ensemble storage'], row['path']

    @classmethod
    def of(cls, row):

        cooked = cls.prep(row)
        S = cooked['action']
        G = cooked['generator']
        last = cooked['thermalization'].configuration[-1]

        E = supervillain.Ensemble(S).generate(row['configurations'], G, start=last, progress=progress)

        E.measure()
        try:
            tau = E.autocorrelation_time()
            logger.info(f'Production τ={tau}')
        except Exception as exception:
            logger.warning(exception)
            logger.warning(f'Setting τ to 2, assuming the thermalization decorrelated correctly.')
            tau=2
        E.tau = tau

        return E

# Finally we bootstrap, for later analysis and plotting.
@h5_cached
class Bootstrap(Step):

    ingredients = {
            'ensemble': Ensemble,
            }

    @classmethod
    def target(cls, row):
        return row['bootstrap storage'], row['path']

    @classmethod
    def of(cls, row):

        cooked = cls.prep(row)
        E = cooked['ensemble']

        return supervillain.analysis.Bootstrap(E.every(E.tau), row['bootstraps'])

