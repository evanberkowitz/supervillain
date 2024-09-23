#!/usr/bin/env python

import supervillain

parser = supervillain.cli.ArgumentParser()
args = parser.parse_args()

print(supervillain.meta.header)
print(f'{supervillain.meta.authors:>80s}')

v = f'Version {supervillain.meta.version} {supervillain.meta.version_name}'
print(f'{v:>80s}')
