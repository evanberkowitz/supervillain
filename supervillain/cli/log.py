#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

import argparse

####
#### SETUP
####

# The aim is to allow the user to add different flags to an ArgumentParser relatively simply,
# with the syntactic suger
#
#   parser.add_argument('--option',  **TheOption())
#
# which proved to be a decent design choice for https://github.com/evanberkowitz/argparse-numpy-slices/
# This allows the user to pick different flag names than defaults() below but still leverage
# the actions provided here.
#
# An object which can be **unpacked into ArgumentParser.add_argument is of type StarStarSugar.
#
class StarStarSugar:
    # The idea is that TheOption should inherit from StarStarSugar
    def __init__(self, **kwargs):
        # and provide a default set of parameters, that can be overridden
        # when the object is created.
        self.parameters.update({**kwargs})
        if 'default' in self.parameters:
            self.parameters['help']+=f' Default is {str(self.parameters["default"]).replace("%","%%")}.'
        # All of the "meat" is that the action should be a custom argparse.Action
        #
        #   https://docs.python.org/3/library/argparse.html#argparse.Action
        #
        # which is accomplished by creating anonymous classes on the fly,
        # via the LogAction function, below.
        self.parameters['action'] = LogAction(self)

    # By Providing keys
    def keys(self):
        return self.parameters.keys()
    # and __getitem__
    def __getitem__(self, key):
        return self.parameters[key]
    # we make sure that we can **unpack StarStarSugar objects into parser.add_argument.
    def method(self, values):
        # Each child class needs to provide their own method.
        raise NotImplemented()

def LogAction(action):
    '''Construct an argparse.Action on the fly with init and call determined by the passed action.'''

    # Here is where we actually inherit from argparse.Action!
    class anonymous(argparse.Action):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # On construction, set the default value.
            logging.basicConfig()
            action.method(action['default'])

        def __call__(self, parser, namespace, values, option_string, ):
            # On parsing, set the passed value.
            action.method(values)

    return anonymous

####
#### ACTUAL LOGGING OPTIONS
####

# Now we can begin implementing some options.
#
# First, something to parse logging levels
#
#   https://docs.python.org/3/library/logging.html#logging-levels
#
class LogLevel(StarStarSugar):

    parameters = {
            'default': 'WARNING',
            'help':    'Log level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL.',
            'type':    str
            }

    levels = {
            'DEBUG':    logging.DEBUG,
            'INFO':     logging.INFO,
            'WARNING':  logging.WARNING,
            'ERROR':    logging.ERROR,
            'CRITICAL': logging.CRITICAL,
            }

    def method(self, values):
        try:
            logging.getLogger().setLevel(self.levels[values])
        except:
            raise ValueError(f'Must be one of {self.levels.keys()}')

# Next, something to control the log format.
# The default is a custom format with a good balance of information.
class LogFormat(StarStarSugar):

    parameters = {
            'type': str,
            'default': '%(asctime)s %(name)-30s %(levelname)10s %(message)s',
            'help': 'Log format.  See https://docs.python.org/3/library/logging.html#logrecord-attributes for details.',
            }

    def method(self, fmt):
        formatter= logging.Formatter(fmt)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

def defaults():
    r'''
    Returns an ``ArgumentParser`` which includes
    
    * ``--log-level``
    * ``--log-format``

    When parsed, even with the default arguments, these options trigger a call to `logging.basicConfig`_.
    If you want to manually control the logging setup you can override the created configuration by specifying ``force=True``.

    .. _logging.basicConfig: https://docs.python.org/3/library/logging.html#logging.basicConfig
    '''
    log_arguments = argparse.ArgumentParser(add_help=False)
    log_arguments.add_argument('--log-level',  **LogLevel())
    log_arguments.add_argument('--log-format', **LogFormat())
    return log_arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[defaults(), ]) 
    args = parser.parse_args()
    print(args)

    logger.debug    ("This is DEBUG")
    logger.info     ("This is INFO")
    logger.warning  ("This is a WARNING")
    logger.error    ("This is an ERROR")
    logger.critical ("This is CRITICAL")
