import supervillain.meta
from supervillain.license import copyright, license
import argparse

# Every metadata flag will print something and then cause the program to exit.
#
#   parser.add_argument('--option',  **print_and_exit(description, data))
#
# where if the user calls
# --help they see the description
# --option they see the data.
def print_and_exit(description, data):

    # We construct an anonymous argparse.Action
    #
    #   https://docs.python.org/3/library/argparse.html#argparse.Action
    #
    class anonymous(argparse.Action):
    # which, when called
        def __call__(self, parser, namespace, values, option_string):
            # prints and exits.
            if callable(data):
                print(data())
            else:
                print(data)
            exit()

    # Return a dictionary which can be **unpacked.
    return {
            'help':   f'Exit after printing {description}.',
            'action': anonymous,
            'nargs':  0,
            }

def defaults():
    r'''
    Returns an ``ArgumentParser`` which includes
    
    * ``--version``
    * ``--copyright``
    * ``--license``

    These options print some information about supervillain itself and then cause the program to exit.
    '''
    meta_arguments = argparse.ArgumentParser(add_help=False)
    meta_arguments.add_argument('--version', **print_and_exit("the version", supervillain.meta.version))
    meta_arguments.add_argument('--copyright', **print_and_exit("copyright information", copyright))
    meta_arguments.add_argument('--license', **print_and_exit("the license", license))

    return meta_arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[defaults(), ]) 
    args = parser.parse_args()

