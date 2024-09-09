#!/usr/bin/env python

import supervillain
from supervillain.performance import Timer
from steps import Thermalization

import logging
logger = logging.getLogger(__name__)

def produce(ensembles):

    with Timer(logger.info, f'Thermalizng {len(ensembles)} ensembles'):
        for idx, row in ensembles.sort_values(by=['W', 'N', 'kappa'], ascending=True).iterrows():

            for line in str(row).split('\n'):
                logger.info(line)

            with Timer(logger.info, f'Thermalizing for W={row["W"]} N={row["N"]} κ={row["kappa"]:0.6f}'):
                B = Thermalization.of(row)

if __name__ == '__main__':

    parser = supervillain.cli.ArgumentParser()
    parser.add_argument('input_file', type=supervillain.cli.input_file('input'), default='input.py')
    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--parallel-files', default=False, action='store_true', help='Store not in the usual storage spots but instead where it would be stored in a --parallel computation.  Useful for testing / debugging.')

    args = parser.parse_args()

    if not args.parallel:
        if args.parallel_files:
            from parallel import io_prep
            args.input_file.ensembles = args.input_file.ensembles.apply(io_prep, axis=1)

        from tqdm.autonotebook import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm
        import steps
        steps.progress=tqdm

        with logging_redirect_tqdm():
            produce(args.input_file.ensembles)
    else:
        from parallel import Parallelize
        Parallelize(produce)(args.input_file.ensembles, gather=('thermalization storage', ))
