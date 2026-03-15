"""quartz (alfa-quartz) 
G. Ghosh., Opt. Commun. 163, 95-102 (1999)"""

import numpy as np
from numpy.lib.scimath import sqrt
from numpy.linalg import inv
Pi=np.pi

lambda_min=0.198
lambda_max=2.053

def n(x):
    no=(1+0.28604141+1.07044083/(1-1.00585997e-2/x**2)+1.10202242/(1-100/x**2))**.5
    ne=(1+0.28851804+1.09509924/(1-1.02101864e-2/x**2)+1.15662475/(1-100/x**2))**.5
    return np.array([no,ne])

"""nonlinear dispersion"""