from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import supervillain
from functools import cache

import logging
logger = logging.getLogger(__name__)

@cache
def Lattice(N):
    return supervillain.Lattice2D(N)

def villain(configurations, N, kappa, W=1):
    L = Lattice(N)
    S = supervillain.action.Villain(L, kappa, W=W)

    if W == 1:
        G = supervillain.generator.combining.Sequentially((
                supervillain.generator.villain.SiteUpdate(S),
                supervillain.generator.villain.LinkUpdate(S),
                supervillain.generator.villain.ExactUpdate(S),
                supervillain.generator.villain.HolonomyUpdate(S),
            ))
    else:
        G = supervillain.generator.combining.Sequentially((
                supervillain.generator.villain.SiteUpdate(S),
                supervillain.generator.villain.LinkUpdate(S),
                supervillain.generator.villain.ExactUpdate(S),
                supervillain.generator.villain.HolonomyUpdate(S),
                # Can also make Δn=±1 changes by the worm in a dn=0 way.
                supervillain.generator.villain.worm.Geometric(S),
            ))

    with logging_redirect_tqdm():
        ensemble = supervillain.Ensemble(S).generate(configurations, G, start='cold', progress=tqdm)

    return ensemble

def worldline(configurations, N, kappa, W=1):
    L = Lattice(N)
    S=supervillain.action.Worldline(L, kappa, W=W)
    G = supervillain.generator.combining.Sequentially((
        supervillain.generator.worldline.VortexUpdate(S),
        supervillain.generator.worldline.CoexactUpdate(S),
        supervillain.generator.worldline.WrappingUpdate(S, 1),
        supervillain.generator.worldline.worm.Geometric(S),
        ))

    with logging_redirect_tqdm():
        ensemble = supervillain.Ensemble(S).generate(configurations, G, start='cold', progress=tqdm)
    return ensemble

def ensemble(configurations, action, N, kappa, W=1):
    if action == 'villain':
        return villain(configurations, N, kappa, W)
    elif action == 'worldline':
        return worldline(configurations, N, kappa, W)
    raise ValueError('Unknown action')

@cache
def cached_ensemble(action, configurations, N, kappa, W=1):
    return ensemble(configurations, action, N, kappa, W)

