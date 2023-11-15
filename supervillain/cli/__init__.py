import argparse
import supervillain.meta

import logging
logger = logging.getLogger(__name__)

from .log import defaults as log_defaults
from .metadata import defaults as meta_defaults

def defaults():
    r'''
    Provides a list of standard-library ``ArgumentParser`` objects.

    Currently provides defaults from

    * :func:`supervillain.cli.log.defaults`
    * :func:`supervillain.cli.metadata.defaults`
    '''
    return [
            log_defaults(),
            meta_defaults()
            ]

class ArgumentParser(argparse.ArgumentParser):
    r'''
    Forwards all arguments, except that it adds :func:`~.cli.defaults` to the `parents`_ option.

    Parameters
    ----------
        *args:
            Forwarded to the standard library's ``ArgumentParser``.
        *kwargs:
            Forwarded to the standard library's ``ArgumentParser``.

    .. _parents: https://docs.python.org/3/library/argparse.html#parents
    '''
    def __init__(self, *args, **kwargs):
        k = {**kwargs}
        if 'parents' in k:
            k['parents'] += defaults()
        else:
            k['parents'] = defaults()
        super().__init__(*args, **k, epilog=f'Built on the supervillain library from {supervillain.meta.authors}.')

    def parse_args(self, args=None, namespace=None):
        r'''
        Forwards to the `standard library`_ but logs all the parsed values at the `DEBUG` level.

        .. _standard library: https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        '''

        parsed = super().parse_args(args, namespace)

        for arg in parsed.__dict__:
            logger.debug(f'{arg}: {parsed.__dict__[arg]}')

        return parsed
