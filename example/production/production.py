#!/usr/bin/env python

import supervillain
from supervillain.performance import Timer
from steps import Bootstrap

import logging
logger = logging.getLogger(__name__)

def produce(ensembles):

    with Timer(logger.info, f'Producing {len(ensembles)} bootstraps'):
        for idx, row in ensembles.sort_values(by=['W', 'N', 'kappa'], ascending=True).iterrows():

            for line in str(row).split('\n'):
                logger.info(line)

            with Timer(logger.info, f'Producing bootstrap for W={row["W"]} N={row["N"]} κ={row["kappa"]:0.6f}'):
                B = Bootstrap.of(row)

if __name__ == '__main__':

    parser = supervillain.cli.ArgumentParser()
    parser.add_argument('input_file', type=supervillain.cli.input_file('input'), default='input.py')

    args = parser.parse_args()

    produce(args.input_file.ensembles)