"""Calcite 
G. Ghosh., Opt. Commun. 163, 95-102 (1999)"""

import numpy as np
from numpy.lib.scimath import sqrt
from numpy.linalg import inv
Pi=np.pi

lambda_min=0.2
lambda_max=2.172

def n(x):
    no=(1+0.73358749+0.96464345/(1-1.94325203e-2/x**2)+1.82831454/(1-120/x**2))**.5
    ne=(1+0.35859695+0.82427830/(1-1.06689543e-2/x**2)+0.14429128/(1-120/x**2))**.5
    return np.array([no,ne])

"""nonlinear dispersion"""