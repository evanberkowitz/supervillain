

class Generator:

    def step(self, configuration):
        r'''
        Parameters
        ----------
            configuration:  dictionary
                Contains (at least) the relevant field variables from which to update the Markov chain.
        Returns
        -------
            dictionary
                The next configuration (and any inline observables), also as a dictionary.
        '''

        return x.copy()

    def inline_observables(self, steps):
        r'''
        Parameters
        ----------
            steps: int

        Returns
        -------
            dict
                Containing an initialized set of inline observables,
                each with an outer dimension of size ``steps``.
                Presumably it makes sense to initialize to 0, but it's not required.
        '''

        return dict()
