#!/usr/bin/env python

import supervillain

parser = supervillain.cli.ArgumentParser()
parser.add_argument('--N', type=int, default=5, help='Sites on a side.')
parser.add_argument('--kappa', type=float, default=0.5, help='κ.')
parser.add_argument('--W', type=int, default=1, help='Winding constraint W.')
parser.add_argument('--configurations', type=int, default=10000)
parser.add_argument('--independent-samples', type=int, default=1000)
parser.add_argument('--action', type=str, default='villain', choices=['villain', 'worldline'])

h5 = 'example.h5'

def path(args, prefix=''):
    return prefix + f'/kappa={args.kappa:0.5f}/W={args.W}/N={args.N}/{args.action}'
def file(args, ext='.pdf'):
    return f'kappa={args.kappa:0.5f}-W={args.W}-N={args.N}-{args.action}' + ext

def ensemble(args):
    return path(args, '/ensemble')

def decorrelated(args):
    return path(args, '/decorrelated')

def bootstrap(args):
    return path(args, '/bootstrap')

if __name__ == '__main__':

    print('''
    In this small production example, different scripts take the same set of arguments.
    All the data is written to and read from example.h5 and some paths are hard-coded.
    In a real production calculation it is probably better to have finer-grained storage.

    generate.py
        - Generates a Monte Carlo stream,
        - Measures observables,
        - Decorrelate using the autocorrelation time.

    plot-history.py
        Visualizes observables of the original and decorrelated streams.

    bootstrap.py
        Resamples the decorrelated stream.

    reset.py
        Remove the streams and bootstrap.

    All of the scripts above understand the same arguments, shown below this introduction.
    They operate on one ensemble at a time.  So, as an example, one may do

        ./generate.py
        ./bootstrap.py
        ./plot-history.py

    to generate, bootstrap, and visualize the default parameters.

    The compare.py script takes paths to bootstrap resamplings, evalutes their observables' bootstrap estimates, and shows them as a table.  So, after the default ensembles are generated you can generate an ensemble with the same parameters but in the worldline formuation with

        ./generate.py --action worldline
        ./bootstrap.py --action worldline
        ./plot-history.py --action worldline

    after which we can

        ./compare.py /bootstrap/kappa=0.50000/W=1/N=5/villain /bootstrap/kappa=0.50000/W=1/N=5/worldline

    Rather than do all of this manually, end-to-end.py shows how we can use the functionality from each of these scripts to operate on many ensembles at once.
    ''')

    parser.parse_args(['--help'])
