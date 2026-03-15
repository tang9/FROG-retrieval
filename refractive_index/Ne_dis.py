"""
 Appl. Opt. 47, 4856-4863 (2008)
"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12


lambda_min=0.4
lambda_max=1

def n(x):  
    return (1+9154.48e-8/(1-656.97e-6/x**2)+4018.63e-8/(1-5.728e-3/x**2))**.5