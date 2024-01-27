#!/usr/bin/env python

import numpy as np
import supervillain.action
from supervillain.h5 import ReadWriteable

import logging
logger = logging.getLogger(__name__)

#Random start
#New direction randomly
#Move worm

#Try larger kappas?

class SimpleWorm(ReadWriteable):
    pass
#?

class SlowUndirectedWorm(ReadWriteable):
    r'''
    Same algorithm as UndirectedWorm, but with logging capabilities to help with statistics management and diagnostics.
    '''

    def __init__(self, action, interval_m = 1):

        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The (undirected) worm algorithm update requires the Worldline action.')
        self.Action     = action
        self.kappa      = action.kappa
        self.Lattice    = action.Lattice

        #In case we want to implement a generalized algorithm in the future.
        #Unimportant now
        self.interval_m = interval_m

        self.rng = np.random.default_rng()

        #Debugging lists to diagnose a worm ensemble
        self.startpointlist = []
        self.lastconfigslist = []
        self.endconfiglist = []
        self.endconfiglist_before_erasure = []
        self.lastsiteslist = []
        self.lengthslist = []
        self.lastprobslist = []
        self.erasedlist = []


        #TODO: Implement statistics monitors
        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        #Count of how many worms have been attempted (including erased/trivial worms)
        self.sweeps = 0
        self.total_burrows = 0
        #Counter for total length of all worms
        #Calculated by taking the abs of the worm (difference of new vs old config) and totalling.
        self.total_length = 0
        #Counts total number of accepted, non-zero worms (Not accounting for disconnected steps)
        self.worm_count = 0
        self.erasure_Ns = []
        self.avg_length_all = max(1,self.total_length)/max(1,self.sweeps)
        self.avg_length_accepted = max(1,self.total_length)/max(1,self.accepted)
        self.avg_length_nontriv = max(1,self.total_length)/max(1,self.worm_count)

    def burrow(self, cfg, currentsite):
        # Takes in the current position of the worm, returns the new configuration after the worm has moved and the coordinate to which it has burrowed.
        pT, mT, mX, pX = self.Lattice.mod(currentsite + np.array([[+1,0],[-1,0],[0,-1],[0,+1]]))
        
        # +T
        proposedconfigpT = cfg.copy()
        #The value of the link that is being traversed
        currentlink = proposedconfigpT[0,currentsite[0],currentsite[1]]
        #When travelling along the positive axis, increase the link value
        proposedlink = currentlink + 1
        #Difference in Action (Energy in literature) between proposition and current state
        deltaSpT = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASpT = min(1,np.exp(-deltaSpT))

        # -T
        proposedconfigmT = cfg.copy()
        currentlink = proposedconfigmT[0,mT[0],mT[1]]
        proposedlink = currentlink - 1
        deltaSmT = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASmT = min(1,np.exp(-deltaSmT))
        
        #+X
        proposedconfigpX = cfg.copy()
        currentlink = proposedconfigpX[1,currentsite[0],currentsite[1]]
        proposedlink = currentlink + 1
        deltaSpX = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASpX = min(1,np.exp(-deltaSpX))
        
        #-X
        proposedconfigmX = cfg.copy()
        currentlink = proposedconfigmX[1,mX[0],mX[1]]
        proposedlink = currentlink - 1
        deltaSmX = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASmX = min(1,np.exp(-deltaSmX))

        #Normalization for the weighting of 4 directons
        N = np.sum(np.array([ASpT,ASmT,ASpX,ASmX]))
        #I was running into errors where the normalization was 0, so this was to help debug that
        #If anyone has suggestions about how to better do this using a logger please let me know
        if(N == 0):
            print('Error')
            print([ASpT,ASmT,ASpX,ASmX])
            print([deltaSpT,deltaSmT,deltaSpX,deltaSmX])
            print(currentsite)
            raise ValueError('N is 0')
        
        #The weights for each direction
        #+T
        PpT = ASpT/N
        #-T
        PmT = ASmT/N
        #+X
        PpX = ASpX/N
        #-X
        PmX = ASmX/N

        chosenDirection = self.rng.choice([1,2,3,4],p=[PpT,PmT,PpX,PmX])
        #1=PpT=+T, 2=PmT=-T, 3=PpX=+X, 4=PmX=-X
        newconfig = cfg.copy()

        #+T
        if chosenDirection==1:
            #For movement with +T, increase the link
            newconfig[0,currentsite[0],currentsite[1]] += 1
            #Updates the location of the head of the worm
            newsite = pT
        #-T
        if chosenDirection==2:
            newconfig[0,mT[0],mT[1]] -= 1
            newsite = mT
        #+X
        if chosenDirection==3:
            newconfig[1,currentsite[0],currentsite[1]] += 1
            newsite = pX
        #-X
        if chosenDirection==4:
            newconfig[1,mX[0],mX[1]] -= 1
            newsite = mX

        return [newconfig,newsite,[PpT,PmT,PpX,PmX],N]

    def step(self, cfg):
        # Step encapsulates one iteration of a worm to completion, either with acceptance or erasure
        
        # Structure: 
        #   -Starting site chosen
        #       ? Eventually may want to store starting sites
        #       *No subroutine just initialized at start of each step
        #   -Choose direction to move, as per metropolis weighting
        #       -Involves calculating weights for each of 4 links
        #       *This will repeat until the worm has returned to start. Looping call to subroutine 'burrow'. 
        #       *Weightings can be calculated within burrow to start, may be relegated to inner subroutine later.
        #       -End of burrow subroutine returns starting config with one link changed
        #           -May include records of directional biases later, so will store output in dict
        #       -Final check for worm erasure, 
        #       -After burrow concludes loop, add to avg_length
        #       !!!!! Ask about whether erased worms should contribute to the length records. Will not to begin with

        #At each step, randomly choose a first site for the worm
        startpoint = self.rng.choice(self.Lattice.coordinates)
        self.startpoint = startpoint
        self.startpointlist.append(startpoint)
        #Initializing value for endpoint so that the loop runs
        endpoint = np.empty_like(startpoint)
        #The 'head' of the worm
        currentpoint = startpoint.copy()
        #Storage for each worm's history
        configlist = []
        probslist = []
        siteslist = []
        
        burrows = 0
        length = 0

        currentconfig = cfg['m']
        initconfig = currentconfig.copy()
        while (endpoint != startpoint).any():
            newconfig, endpoint, probs, Ns = self.burrow(currentconfig, currentpoint)
            #Updates position of the head of the worm
            currentpoint = endpoint.copy()
            currentconfig = newconfig.copy()
            configlist.append(currentconfig)
            siteslist.append(endpoint)
            probslist.append(probs)
            burrows += 1
            self.lastsiteslist = siteslist
            self.lastconfigslist = configlist
            self.lastprobslist = probslist
            

        #Calculates the total length of all worm(s) proposed in a step
        length = np.sum(np.abs(currentconfig-initconfig))
        self.lengthslist.append(length)
        self.endconfiglist_before_erasure.append(currentconfig)
        self.total_burrows += burrows
        #Worm erasure step
        #N_no_worm
        _,_,_,nwN = self.burrow(cfg['m'],startpoint)
        _,_,_,wN = self.burrow(currentconfig,startpoint)
        self.erasure_Ns.append([nwN,wN])
        erasure_metropolis = self.rng.uniform(0,1)
        erasure_prob = 1-min([1,nwN/wN])
        self.acceptance += 1-erasure_prob
        if(erasure_metropolis <= erasure_prob):
            currentconfig = cfg['m'].copy()
            logger.debug('Worm erased')
            self.erasedlist.append(True)
            if length != 0:
            #TODO: For steps that create two disconnected worms, it may be necessary to count each separately. Pending implementation
                self.worm_count += 1
        else:
            logger.debug('Worm not erased')
            self.accepted += 1
            self.erasedlist.append(False)
            self.total_length += length
            if length != 0:
            #TODO: For steps that create two disconnected worms, it may be necessary to count each separately. Pending implementation
                self.worm_count += 1
        logger.debug(f'Burrows: {burrows}')
        logger.debug(f'Length: {length}')
        current = {}
        self.sweeps += 1

        #Updates average length
        self.avg_length_all = max(1,self.total_length)/max(1,self.sweeps)
        self.avg_length_accepted = max(1,self.total_length)/max(1,self.accepted)
        self.avg_length_nontriv = max(1,self.total_length)/max(1,self.worm_count)
        current['m'] = currentconfig
        self.endconfiglist.append(currentconfig)
        return current
    
    def report(self):
        r'''
        Returns a string with some summarizing statistics.
        '''
        return (
            f'There were {self.accepted} worms accepted of {self.sweeps} attempted worms.'
            +'\n'+
            f'    {self.accepted/self.sweeps:.6f} acceptance rate.' 
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
            +'\n'+
            f'    {self.avg_length_all:.6f} average length over all worm attempts'
            +'\n'+
            f'    {self.avg_length_accepted:.6f} average length over all ACCEPTED worm attempts'
            +'\n'+
            f'    {self.avg_length_nontriv:.6f} average length over all ACCEPTED, NON-ZERO worm attemps'
            +'\n'+
            f'    {self.total_burrows} total burrows'
        )

class UndirectedWorm(ReadWriteable):
    r'''
    Ref. :cite:`PhysRevE.67.015701` gives us an ergotic and balanced worm algorithm in the context of the quantum rotor model. The quantum rotor model is of a similar class to our worldline formulation, requiring divergenceless currents, so the implementation of this worm is straightforward.
    The meat and potatos of this algorithm are contained within the :func:`~burrow` and :func:`~step` methods, with the step method repeatedly referring to burrow.
    
    Much like every other generator explained thus far, the worm updates in steps, where in this case a step corresponds to a closed-loop worm. 
    It is important that these worms are closed, for this preserves the value of $\delta m$ throughout the lattice. For the :class:`~.Worldline` action, we aim to sample only configurations with $\delta m = 0$.
    A worm is generated through the repeated use of :func:`~burrow`. Starting from a random location on the lattice, a :func:`~burrow` is performed in one of the four directions from the stencil, where the direction traveled depends on the action difference. Burrows are then continually performed until the 'head' of the worm returns to its 'tail'. At this point the worm is then tested for "erasure" to preserve detailed balance. Erasure probabilities remain low and have little impact on the efficiency of the algorithm, since worms are rarely erased.
    

    According to Ref. :cite:`PhysRevE.67.015701`, this algorithm gives a power law relation between autocorrelation times and lattice size whereas a scheme similar to our combined :class:`~.PlaquetteUpdate` and :class:`~.HolonomyUpdate` gives either an exponentially increasing relationship, or a power law of notably higher degree.
    Our specific tests show that the worm (when accounting for autocorrelation times due to the *wrapping* holonomies) reduces the explosion of autocorrelation times dramatically, particularly with larger lattices.
    Even near critical $\kappa$ autocorrelation times remain on the order of 10s for $L=50$ lattices, whereas traditional algorithms yield autocorrelation times on the order of 100s.
    '''

    def __init__(self, action):

        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The (undirected) worm algorithm update requires the Worldline action.')
        self.Action     = action
        self.kappa      = action.kappa
        self.Lattice    = action.Lattice

        self.rng = np.random.default_rng()

        #TODO: Implement statistics monitors
        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        #Count of how many worms have been attempted (including erased/trivial worms)
        self.sweeps = 0
        #Counter for total length of all worms
        #Counts total number of accepted, non-zero worms (Not accounting for disconnected steps)
        self.worm_count = 0

    def burrow(self, cfg, currentsite):
        r'''
        Each burrow step moves the 'head' of the worm to a semi-random neighboring lattice site weighted according to the change in action generated by that move.
        A single burrow of the worm in the positive X or T direction will increment the value of the $m$ field that lives on the link it moves across by 1, and likewise decrement $m$ when it moves in the negative X or T direction.
        The probability of moving in a direction is given by $\frac{A_l}{N}$ where
        $$ 
        A_{l}^\sigma=\min(1,e^{-\Delta S_{l}^\sigma/\kappa})
        $$
        and $N = \sum_{l} A_{l} $ for each link $l$ in the stencil around the current head of the worm.

        Parameters
        ----------
            cfg : np.ndarray    
                configuration to burrow in
            currentsite : 
                coordinates on the array indicating the head of the worm
        Returns
        -------
            an array of results from each burrow
                - The new link configuration after the burrow: np.ndarray
                - The coordinates of the new head of the worm: np.ndarray
                - An array of the probabilities calculated for each potential burrow direction [+T,-T,+X,-X]: Python array
                - The normalization N of this burrow step: int

        '''
        # Takes in the current position of the worm, returns the new configuration after the worm has moved and the coordinate to which it has burrowed.
        pT, mT, mX, pX = self.Lattice.mod(currentsite + np.array([[+1,0],[-1,0],[0,-1],[0,+1]]))
        
        # +T
        proposedconfigpT = cfg.copy()
        #The value of the link that is being traversed
        currentlink = proposedconfigpT[0,currentsite[0],currentsite[1]]
        #When travelling along the positive axis, increase the link value
        proposedlink = currentlink + 1
        #Difference in Action (Energy in literature) between proposition and current state
        deltaSpT = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASpT = min(1,np.exp(-deltaSpT))

        # -T
        proposedconfigmT = cfg.copy()
        currentlink = proposedconfigmT[0,mT[0],mT[1]]
        proposedlink = currentlink - 1
        deltaSmT = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASmT = min(1,np.exp(-deltaSmT))
        
        #+X
        proposedconfigpX = cfg.copy()
        currentlink = proposedconfigpX[1,currentsite[0],currentsite[1]]
        proposedlink = currentlink + 1
        deltaSpX = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASpX = min(1,np.exp(-deltaSpX))
        
        #-X
        proposedconfigmX = cfg.copy()
        currentlink = proposedconfigmX[1,mX[0],mX[1]]
        proposedlink = currentlink - 1
        deltaSmX = 1/(2*self.kappa)*(proposedlink**2 - currentlink**2)
        ASmX = min(1,np.exp(-deltaSmX))

        #Normalization for the weighting of 4 directons
        N = np.sum(np.array([ASpT,ASmT,ASpX,ASmX]))
        
        #The weights for each direction
        #+T
        PpT = ASpT/N
        #-T
        PmT = ASmT/N
        #+X
        PpX = ASpX/N
        #-X
        PmX = ASmX/N

        chosenDirection = self.rng.choice([1,2,3,4],p=[PpT,PmT,PpX,PmX])
        #1=PpT=+T, 2=PmT=-T, 3=PpX=+X, 4=PmX=-X
        newconfig = cfg.copy()

        #+T
        if chosenDirection==1:
            #For movement with +T, increase the link
            newconfig[0,currentsite[0],currentsite[1]] += 1
            #Updates the location of the head of the worm
            newsite = pT
        #-T
        if chosenDirection==2:
            newconfig[0,mT[0],mT[1]] -= 1
            newsite = mT
        #+X
        if chosenDirection==3:
            newconfig[1,currentsite[0],currentsite[1]] += 1
            newsite = pX
        #-X
        if chosenDirection==4:
            newconfig[1,mX[0],mX[1]] -= 1
            newsite = mX

        return [newconfig,newsite,[PpT,PmT,PpX,PmX],N]

    def step(self, cfg):
        r'''
        :class:`~.Ensembles` uses this method as it does with other generators. 
        Each worm 'step' corresponds to the completion of a worm, ie. the head meeting the tail. This method starts the worm from a random site, repeatedly calls :func:`~burrow`. 
        At the end of a step, worm erasure is determined depending on the worm erasure probability in Ref. :cite:`PhysRevE.67.015701`
        $$ P_{erasure} = 1-\text{min}\left(1,\frac{N}{N_{worm}}\right) $$
        where $N$ represents the value of N calculated in the first worm step and $N_worm$ is the value of N calculated after the worm has finished, from the starting point.
        Statistics such as worm length can be recorded, and in the future may be used to calculate correlations in situ.

        Parameters
        ----------
            cfg: dictionary of starting configuration before the step.

        Returns
        -------
            current: dictionary of the new configuration.
        '''
        # Step encapsulates one iteration of a worm to completion, either with acceptance or erasure
        
        # Structure: 
        #   -Starting site chosen
        #       ? Eventually may want to store starting sites
        #       *No subroutine just initialized at start of each step
        #   -Choose direction to move, as per metropolis weighting
        #       -Involves calculating weights for each of 4 links
        #       *This will repeat until the worm has returned to start. Looping call to subroutine 'burrow'. 
        #       *Weightings can be calculated within burrow to start, may be relegated to inner subroutine later.
        #       -End of burrow subroutine returns starting config with one link changed
        #           -May include records of directional biases later, so will store output in dict
        #       -Final check for worm erasure, 
        #       -After burrow concludes loop, add to avg_length
        #       !!!!! Ask about whether erased worms should contribute to the length records. Will not to begin with

        #At each step, randomly choose a first site for the worm
        startpoint = self.rng.choice(self.Lattice.coordinates)
        #Initializing value for endpoint so that the loop runs
        endpoint = np.empty_like(startpoint)
        #The 'head' of the worm
        currentpoint = startpoint.copy()
        burrows=0
        currentconfig = cfg['m']
        initconfig = currentconfig.copy()
        while (endpoint != startpoint).any():
            newconfig, endpoint, probs, Ns = self.burrow(currentconfig, currentpoint)
            #Updates position of the head of the worm
            currentpoint = endpoint.copy()
            currentconfig = newconfig.copy()
            burrows += 1

        #Calculates the total length of all worm(s) proposed in a step
        length = np.sum(np.abs(currentconfig-initconfig))
        #Worm erasure step
        #N_no_worm
        _,_,_,nwN = self.burrow(cfg['m'],startpoint)
        _,_,_,wN = self.burrow(currentconfig,startpoint)
        erasure_metropolis = self.rng.uniform(0,1)
        erasure_prob = 1-min([1,nwN/wN])
        self.acceptance += 1-erasure_prob
        if(erasure_metropolis <= erasure_prob):
            currentconfig = cfg['m'].copy()
            if length != 0:
            #TODO: For steps that create two disconnected worms, it may be necessary to count each separately
                self.worm_count += 1
        else:
            self.accepted += 1
            if length != 0:
            #TODO: For steps that create two disconnected worms, it may be necessary to count each separately
                self.worm_count += 1
        current = {}
        self.sweeps += 1

        #Updates average length
        current['m'] = currentconfig
        return current
    
    def report(self):
        r'''
        Returns
        -------
            string with some summarizing statistics.
        '''
        return (
            f'There were {self.accepted} single-site proposals accepted of {self.proposed} proposed updates.'
            +'\n'+
            f'    {self.accepted/self.sweeps:.6f} acceptance rate' 
            +'\n'+
            f'    {self.acceptance / self.sweeps:.6f} average Metropolis acceptance probability.'
        )

class GeometricWorm(ReadWriteable):
    def __init__(self, action):
        if not isinstance(action, supervillain.action.Worldline):
            raise ValueError('The Directed Geometric Worm algorithm requires the Worldline action.')
        self.Action     = action
        self.kappa      = action.kappa
        self.Lattice    = action.Lattice

        self.rng = np.random.default_rng()

        #TODO: Implement statistics monitors
        self.accepted = 0
        self.proposed = 0
        self.acceptance = 0.
        #Count of how many worms have been attempted (including erased/trivial worms)
        self.sweeps = 0
        #Counter for total length of all worms
        #Counts total number of accepted, non-zero worms (Not accounting for disconnected steps)
        self.worm_count = 0
    def burrow(self, ):
        pass
