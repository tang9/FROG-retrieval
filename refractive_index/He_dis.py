"""
 Phys. Rev. A 92, 033821 (2015)
"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12


lambda_min=0.09
lambda_max=1

def n(x):  
    return (1+2.16463842e-05/(1+6.80769781e-04/x**2)+2.10561127e-07/(1-5.13251289e-03/x**2)+4.75092720e-05/(1-3.18621354e-03/x**2))**.5

#needs to be something from np

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3():
    """returns chi3(:,:,:,:)
    https://doi.org/10.1007/10134958_36
    """
    # c_xxxx=1.6*10**-22 #m^2/V^2. approximately. might be 2x less
    # c_xxyy=0.6*10**-22 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    #since it is a cubic crystal
    # chi[0][0][0][0]=c_xxxx
    # chi[1][1][1][1]=c_xxxx
    # chi[2][2][2][2]=c_xxxx
    
    # chi[0][0][1][1]=c_xxyy
    # chi[0][1][0][1]=c_xxyy
    # chi[0][1][1][0]=c_xxyy
    # chi[0][0][2][2]=c_xxyy
    # chi[0][2][0][2]=c_xxyy
    # chi[0][2][2][0]=c_xxyy
    # chi[1][1][0][0]=c_xxyy
    # chi[1][0][1][0]=c_xxyy
    # chi[1][0][0][1]=c_xxyy
    # chi[1][1][2][2]=c_xxyy
    # chi[1][2][1][2]=c_xxyy
    # chi[1][2][2][1]=c_xxyy
    # chi[2][2][0][0]=c_xxyy
    # chi[2][0][2][0]=c_xxyy
    # chi[2][0][0][2]=c_xxyy
    # chi[2][2][1][1]=c_xxyy
    # chi[2][1][2][1]=c_xxyy
    # chi[2][1][1][2]=c_xxyy
    
    return chi