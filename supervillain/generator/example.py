#!/usr/bin/env python

class DoNothing:

    def __init__(self):
        pass

    def step(self, x):
        return {k: v for k, v in x.items()}
