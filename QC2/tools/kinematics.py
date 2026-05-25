import numpy as np

import math
import cmath
from luescher.tools.zeta import Z

#~~~~~~~~~~~~~~~~~~~~~~~`
# energy-dependent kinematic functions
# required for analysis
# ~~~~~
#  convert PSQ from hdf5 to momentum vector
def momentum_state(psq):
        # d = [0,0,0]
        if psq == 'PSQ0':
            return np.array([0,0,0])
        # d = [0,0,1]
        elif psq == 'PSQ1':
            return np.array([0,0,1])
        # d = [1,1,0]
        elif psq == 'PSQ2':
            return np.array([1,1,0])
        # d = [1,1,1]
        elif psq == 'PSQ3':
            return np.array([1,1,1])
        # d = [0,0,2]
        elif psq == 'PSQ4':
            return np.array([0,0,2])
        else: 
            # Raise an exception for invalid input
            raise ValueError("Invalid value for 'i'. 'i' must be 0, 1, 2, or 3.")
# q^2 is COM momentum

def q2(ecm,ma,mb): #assume all ref inputs
    #ecm is the energy data
    # assumed to be (e_cm/m_ref)
    #ma,mb comes from the energies in the channel
    q2 = ecm**2 / 4 - (ma**2 + mb**2) / 2 + ((ma**2 - mb**2)**2) / (4*ecm**2)
    return q2

def q2toecm(q2,ma,mb):
    return np.sqrt(q2 + ma**2) + np.sqrt(q2 + mb**2)
    
def partialE_partialq(q2,ma,mb):
    return np.sqrt(-q2)*(-q2 + ma**2 )**(-1/2) + np.sqrt(-q2)*(-q2 +mb)**(-1/2)

# msplit required for Luscher: 
# measures the shift of the rvector of unequal relativistic masses
def msplit(ecm,ma,mb): #if ma,mb are degenerate its 1
    if ma == mb:
        return 1 
    else:
        return 1 + ((ma**2 - mb**2)/ecm**2 ) 


# ~~~~~~~~~~~~~~~~~~~~~~~`
# lorentz factor
# gamma = E/E^*`
# ref is reference mass from data set
# d is PSQ of the state
def gamma(ecm,d,L,ref):
    d_vec = momentum_state(d) 
    L_ref = L*ref
    E = math.sqrt(ecm**2 + (((2*math.pi)/L_ref)**2)*np.dot(d_vec,d_vec))
    #print("E=",E)
    return E/ ecm #np.abs(ecm)

# Leuscher Zeta Function
# using numerical approach including numerical integrals
def qcotd(ecm,L,psq,ma,mb,ref):
    if ecm < 0:
        return np.inf
    L_ref = L*ref
    d_vec = momentum_state(psq) #0,1,2,3
    c = 2 / (gamma(ecm,psq,L,ref)*L_ref*math.sqrt(math.pi))
    #print("c=",c)
    if gamma(ecm,psq,L,ref) < 1: #because energy guess is negative
        return np.abs(gamma)
    #print( psq )
    #print( ma )
    return c*Z(q2(ecm,ma,mb)*((L_ref/(2*math.pi))**2),gamma=gamma(ecm,psq,L,ref),l=0,m=0,d=d_vec,m_split=msplit(ecm,ma,mb),precision=1e-11).real
    